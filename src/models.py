"""
Models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fft
import lie_tools
import utils
import mask

log = utils.log


class CryoFIRE(nn.Module):
    def __init__(self, lattice, output_mask, shared_cnn_params, conf_regressor_params, hyper_volume_params,
                 no_trans=False, sym_loss=True, sym_loss_factor=4, use_gt_poses=False):
        """
        lattice: Lattice
        output_mask: Mask
        shared_cnn_params: dict
            depth_cnn: int
            channels_cnn: int
            kernel_size_cnn: int
            mask_type: str
        conf_regressor_params: dict
            z_dim: int
            std_z_init: float
            variational: bool
        hyper_volume_params: dict
            n_layers: int
            hidden_dim: it
            pe_type: str
            pe_dim: int
            feat_sigma: float
            domain: str
        no_trans: bool
        sym_loss: bool
        sym_loss_factor: bool
        use_gt_poses: bool
        """
        super(CryoFIRE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.output_mask = output_mask

        self.sym_loss = sym_loss
        self.no_trans = no_trans
        self.z_dim = conf_regressor_params['z_dim']
        self.variational_conf = conf_regressor_params['variational']

        # shared cnn
        self.shared_cnn = SharedCNN(self.D, shared_cnn_params['depth_cnn'], shared_cnn_params['channels_cnn'],
                                    shared_cnn_params['kernel_size_cnn'], shared_cnn_params['mask_type'])

        # conformation regressor
        if self.z_dim > 0:
            self.conf_regressor = ConfRegressor(self.shared_cnn.final_channels, self.shared_cnn.final_size,
                                                conf_regressor_params['z_dim'], conf_regressor_params['std_z_init'],
                                                conf_regressor_params['variational'])

        # image duplicator
        if self.sym_loss:
            self.sym_loss_factor = sym_loss_factor
            self.image_duplicator = ImageDuplicator(sym_loss_factor)
        else:
            self.sym_loss_factor = 1

            # pose regressor
        self.use_gt_poses = use_gt_poses
        if not use_gt_poses:
            self.pose_regressor = PoseRegressor(self.shared_cnn.final_channels, self.shared_cnn.final_size, sym_loss,
                                                sym_loss_factor, no_trans)
        else:
            assert not sym_loss, "Symmetric loss must be de-activated when using gt poses"
            log("Will use gt poses")
            self.pose_regressor = GTPoseRegressor(no_trans)

        # hyper-volume
        self.hypervolume = HyperVolume(self.D, self.z_dim, hyper_volume_params['n_layers'],
                                       hyper_volume_params['hidden_dim'],
                                       hyper_volume_params['pe_type'], hyper_volume_params['pe_dim'],
                                       hyper_volume_params['feat_sigma'], hyper_volume_params['domain'])

    def encode(self, in_dict, pose_only):
        """
        in_dict: dict
            y: [batch_size, D, D]
            y_real: [batch_size, D - 1, D - 1]
            R: [batch_size, 3, 3]
            t: [batch_size, 2]
        pose_only: bool

        output: dict
            R: [(sym_loss_factor * ) batch_size, 3, 3]
            t: [(sym_loss_factor * ) batch_size, 2]
            z: [(sym_loss_factor * ) batch_size, z_dim]
            z_logvar: [(sym_loss_factor * ) batch_size, z_dim]
        """
        latent_variables_dict = {}
        shared_features = None
        y_real = in_dict['y_real']
        if self.sym_loss:
            y_real = self.image_duplicator(y_real)
        if self.z_dim > 0 or not self.use_gt_poses:
            shared_features = self.shared_cnn(y_real)
        if self.z_dim > 0:
            conf_dict = self.conf_regressor(shared_features, pose_only)
            for key in conf_dict:
                latent_variables_dict[key] = conf_dict[key]
        pose_dict = self.pose_regressor(shared_features, in_dict)
        for key in pose_dict:
            latent_variables_dict[key] = pose_dict[key]
        return latent_variables_dict

    def decode(self, latent_variables_dict, ctf_local, y_gt):
        """
        latent_variables_dict: dict
            R: [(sym_loss_factor * ) batch_size, 3, 3]
            trans: [(sym_loss_factor * ) batch_size, 2]
            z: [(sym_loss_factor * ) batch_size, z_dim]
            z_logvar: [(sym_loss_factor * ) batch_size, z_dim]
        ctf_local: [batch_size, D, D]
        y_gt: [batch_size, D, D]

        output: [(sym_loss_factor * ) batch_size, n_pts], [batch_size, n_pts]
        """
        batch_size = ctf_local.shape[0]
        rots = latent_variables_dict['R']
        z = None

        # sample conformations
        if self.z_dim > 0:
            if self.variational_conf:
                z = sample_conf(latent_variables_dict['z'], latent_variables_dict['z_logvar'])
            else:
                z = latent_variables_dict['z']

        # generate slices
        x = self.lattice.coords[self.output_mask.binary_mask] @ rots
        y_pred = self.hypervolume(x, z)

        # apply ctf
        if self.sym_loss:
            ctf_local = self.image_duplicator(ctf_local)
        y_pred = self.apply_ctf(y_pred, ctf_local)

        # duplicate gt
        if self.sym_loss:
            y_gt = self.image_duplicator(y_gt)

        # apply translations (to gt)
        if not self.no_trans:
            trans = latent_variables_dict['t'][:, None]
            y_gt_processed = self.lattice.translate_ht(y_gt.reshape(
                self.sym_loss_factor * batch_size, -1), trans).reshape(
                self.sym_loss_factor * batch_size, -1)
        else:
            y_gt_processed = y_gt.reshape(self.sym_loss_factor * batch_size, -1)
        y_gt_processed = y_gt_processed[:, self.output_mask.binary_mask]

        return y_pred, y_gt_processed

    def apply_ctf(self, y_pred, ctf_local):
        """
        y_pred: [(sym_loss_factor * ) batch_size, n_pts]
        ctf_local: [(sym_loss_factor * ) batch_size, D, D]

        output: [(sym_loss_factor * ) batch_size, n_pts]
        """
        ctf_local = ctf_local.reshape(ctf_local.shape[0], -1)[:, self.output_mask.binary_mask]
        y_pred = ctf_local * y_pred
        return y_pred

    def eval_volume(self, norm, zval=None):
        """
        norm: (mean, std)
        zval: [z_dim]
        """
        coords = self.lattice.coords
        extent = self.lattice.extent
        z = None
        if zval is not None:
            assert zval.shape[0] == self.z_dim
            z = torch.tensor(zval, dtype=torch.float32, device=coords.device).reshape(1, self.z_dim)

        volume = np.zeros((self.D, self.D, self.D), dtype=np.float32)
        assert not self.training
        with torch.no_grad():
            for i, dz in enumerate(np.linspace(-extent, extent, self.D, endpoint=True, dtype=np.float32)):
                x = coords + torch.tensor([0, 0, dz], device=coords.device)
                x = x.reshape(1, -1, 3)
                y = self.hypervolume(x, z)
                slice_radius = int(np.sqrt(extent ** 2 - dz ** 2) * self.D)
                slice_mask = mask.CircularMask(self.lattice, slice_radius).binary_mask
                y[0, ~slice_mask] = 0.0
                y = y.view(self.D, self.D).detach().cpu().numpy()
                volume[i] = y
            volume = volume * norm[1] + norm[0]
            volume_real = fft.ihtn_center(volume[0:-1, 0:-1, 0:-1])  # remove last +k freq for inverse FFT
        return volume_real

    @classmethod
    def load(cls, config, weights=None, device=None):
        """
        Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            CryoDRGN3 instance, Lattice instance
        """
        pass


def sample_conf(z_mu, z_logvar):
    """
    z_mu: [batch_size, z_dim]
    z_logvar: [batch_size, z_dim]

    output: [batch_size, z_dim]
    """
    std = nn.Softplus(beta=2)(.5 * z_logvar)
    eps = torch.randn_like(std)
    z = eps * std + z_mu
    return z


class SharedCNN(nn.Module):
    def __init__(self, resolution, depth, channels, kernel_size, mask_type, nl=nn.ReLU):
        """
        resolution: int
        depth: int
        kernel_size: int
        mask_type: str
        """
        super(SharedCNN, self).__init__()

        cnn = []
        in_channels = 1
        out_channels = channels
        final_size = resolution
        for _ in range(depth):
            cnn.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
            in_channels = out_channels
            cnn.append(nl())
            out_channels = 2 * in_channels
            cnn.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
            in_channels = out_channels
            cnn.append(nl())
            cnn.append(nn.AvgPool2d(2))
            final_size = final_size // 2
            cnn.append(nn.GroupNorm(channels, in_channels))
        self.cnn = nn.Sequential(*cnn)

        self.final_size = final_size
        self.final_channels = in_channels

    def forward(self, y_real):
        """
        y_real: [(sym_loss_factor * ) batch_size, D - 1, D - 1]

        output: [(sym_loss_factor * ) batch_size, final_channels, final_size, final_size]
        """
        return self.cnn(y_real[:, None])


class PoseRegressor(nn.Module):
    def __init__(self, channels, kernel_size, sym_loss, sym_loss_factor, no_trans):
        """
        channels: int
        kernel_size: int
        sym_loss: bool
        sym_loss_factor: int
        no_trans: bool
        """
        super(PoseRegressor, self).__init__()
        self.sym_loss = sym_loss
        self.sym_loss_factor = sym_loss_factor
        self.no_trans = no_trans
        self.regressor_rotation = nn.Conv2d(channels, 6, kernel_size, padding='valid')
        if not no_trans:
            self.regressor_translation = nn.Conv2d(channels, 2, kernel_size, padding='valid')

    def forward(self, shared_features, in_dict):
        """
        shared_features: [(sym_loss_factor * ) batch_size, channels, kernel_size, kernel_size]
        in_dict: dict
            y: [batch_size, D, D]
            y_real: [batch_size, D - 1, D - 1]
            R: [batch_size, 3, 3]
            t: [batch_size, 2]

        output: dict
            R: [(sym_loss_factor * ) batch_size, 3, 3]
            t: [(sym_loss_factor * ) batch_size, 2]
        """
        in_size = shared_features.shape[0]
        rots_s2s2 = self.regressor_rotation(shared_features).reshape(in_size, 6)
        rots_matrix = lie_tools.s2s2_to_matrix(rots_s2s2)
        pose_dict = {'R': rots_matrix}
        if not self.no_trans:
            trans = self.regressor_translation(shared_features).reshape(in_size, 2)
            pose_dict['t'] = trans
        return pose_dict


class GTPoseRegressor(nn.Module):
    def __init__(self, no_trans):
        super(GTPoseRegressor, self).__init__()
        self.no_trans = no_trans

    def forward(self, shared_features, in_dict):
        """
        shared_features: [batch_size, channels, kernel_size, kernel_size]
        in_dict: dict
            y: [batch_size, D, D]
            y_real: [batch_size, D - 1, D - 1]
            R: [batch_size, 3, 3]
            t: [batch_size, 2]

        output: dict
            R: [batch_size, 3, 3]
            t: [batch_size, 2]
        """
        pose_dict = {'R': in_dict['R']}
        if not self.no_trans:
            pose_dict['t'] = in_dict['t']
        return pose_dict


class ConfRegressor(nn.Module):
    def __init__(self, channels, kernel_size, z_dim, std_z_init, variational):
        """
        channels: int
        kernel_size: int
        z_dim: int
        std_z_init: float
        variational: bool
        """
        super(ConfRegressor, self).__init__()
        self.z_dim = z_dim
        self.variational = variational
        self.std_z_init = std_z_init
        if variational:
            out_features = 2 * z_dim
        else:
            out_features = z_dim
        self.out_features = out_features
        self.regressor = nn.Conv2d(channels, out_features, kernel_size, padding='valid')

    def forward(self, shared_features, pose_only):
        """
        shared_features: [(sym_loss_factor * ) batch_size, channels, kernel_size, kernel_size]
        pose_only: bool

        output: dict
            z: [(sym_loss_factor * ) batch_size, z_dim]
            z_logvar: [(sym_loss_factor * ) batch_size, z_dim] if variational and not pose_only
        """
        if pose_only:
            in_dim = shared_features.shape[0]
            z = self.std_z_init * torch.randn((in_dim, self.z_dim)).to(shared_features.device)
            conf_dict = {'z': z}
            if self.variational:
                conf_dict['z_logvar'] = torch.ones(
                    (in_dim, self.z_dim), dtype=torch.float32, device=shared_features.device
                )
            return conf_dict
        else:
            z_full = self.regressor(shared_features).reshape(-1, self.out_features)
            if self.variational:
                conf_dict = {
                    'z': z_full[:, :self.z_dim],
                    'z_logvar': z_full[:, self.z_dim:]
                }
            else:
                conf_dict = {
                    'z': z_full
                }
            return conf_dict


class HyperVolume(nn.Module):
    def __init__(self, resolution, z_dim, n_layers, hidden_dim, pe_type, pe_dim, feat_sigma, domain):
        """
        z_dim: int
        n_layers: int
        hidden_dim: int
        pe_type: str
        pe_dim: int
        feat_sigma: float
        domain: str
        """
        super(HyperVolume, self).__init__()
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        if pe_type == "gaussian":
            rand_freqs = torch.randn((3 * pe_dim, 3), dtype=torch.float) * feat_sigma
            self.rand_freqs = nn.Parameter(rand_freqs, requires_grad=False)
        else:
            raise NotImplementedError

        self.D = resolution

        in_features = 3 * 2 * pe_dim + z_dim
        if domain == 'hartley':
            self.mlp = ResidualLinearMLP(in_features, n_layers, hidden_dim, 1)
        else:
            raise NotImplementedError

    def forward(self, x, z):
        """
        x: [(sym_loss_factor * ) batch_size, n_pts, 3]
        z: [(sym_loss_factor * ) batch_size, z_dim] or None

        output: [(sym_loss_factor * ) batch_size, n_pts]
        """
        batch_size_in, n_pts = x.shape[:2]
        if self.pe_type == "gaussian":
            x = self.random_fourier_encoding(x)
        if z is not None:
            x = torch.cat([x, z[:, None].expand(-1, n_pts, -1)], -1)
        return self.mlp(x).reshape(batch_size_in, n_pts)

    def random_fourier_encoding(self, x):
        """
        x: [(sym_loss_factor * ) batch_size, n_pts, 3]

        output: [(sym_loss_factor * ) batch_size, n_pts, 3 * 2 * pe_dim]
        """
        freqs = self.rand_freqs.reshape(1, 1, -1, 3) * (self.D // 2)
        kx_ky_kz = x[..., None, :] * freqs
        k = kx_ky_kz.sum(-1)
        s = torch.sin(k)
        c = torch.cos(k)
        x_encoded = torch.cat([s, c], -1)
        return x_encoded


class ImageDuplicator(nn.Module):
    def __init__(self, sym_loss_factor):
        """
        sym_loss_factor: int
        """
        super(ImageDuplicator, self).__init__()
        assert sym_loss_factor in [2, 4, 8]
        self.sym_loss_factor = sym_loss_factor
        log("Symmetric loss factor is {}".format(sym_loss_factor))

    def forward(self, img):
        """
        img: [batch_size, D, D]

        output: [sym_loss_factor * batch_size, D, D]
        """
        if self.sym_loss_factor == 2:
            # The in-plane rotation described in cryoAI is replaced with an axis-flip.
            # https://arxiv.org/abs/2203.08138
            # img_out = torch.cat([
            #     img[None],
            #     torch.flip(img[None], [-2, -1])
            # ], 0).reshape(-1, *img.shape[-2:])
            img_out = torch.cat([
                img[None],
                torch.flip(img[None], [-1])
            ], 0).reshape(-1, *img.shape[-2:])
        else:
            raise NotImplementedError
        return img_out

    def compensate_transform_rotation(self, rots_full, activated_paths):
        """
        predicted_rots: [sym_loss_factor * batch_size, 3, 3] (numpy)
        activated_paths: [batch_size] (numpy)
        output: [batch_size, 3, 3]
        """
        batch_size = activated_paths.shape[0]
        rots_full = rots_full.reshape(-1, batch_size, *rots_full.shape[1:])
        list_arange = np.arange(batch_size)
        rots = rots_full[activated_paths, list_arange]
        euler_angles = lie_tools.rotmat_to_euler(rots)
        if self.sym_loss_factor == 2:
            path_1 = activated_paths > 0.5

            euler_angles[path_1, 0] = -euler_angles[path_1, 0]
            euler_angles[path_1, 1] = np.pi - euler_angles[path_1, 1]
            euler_angles[path_1, 2] = np.pi + euler_angles[path_1, 2]
        else:
            raise NotImplementedError
        rots = lie_tools.euler_to_rotmat(euler_angles)
        return rots

    def compensate_transform_translation(self, trans_full, activated_paths):
        """
        predicted_rots: [sym_loss_factor * batch_size, 2]
        activated_paths: [batch_size]
        output: [batch_size, 2]
        """
        batch_size = activated_paths.shape[0]
        trans_full = trans_full.reshape(-1, batch_size, *trans_full.shape[1:])
        list_arange = np.arange(batch_size)
        trans = trans_full[activated_paths, list_arange]
        if self.sym_loss_factor == 2:
            trans[activated_paths > 0.5, 0] = -trans[activated_paths > 0.5, 0]
        else:
            raise NotImplementedError
        return trans


class ResidualLinearMLP(nn.Module):
    def __init__(self, in_dim, n_layers, hidden_dim, out_dim, nl=nn.ReLU):
        super(ResidualLinearMLP, self).__init__()
        layers = [ResidualLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim),
                  nl()]
        for n in range(n_layers):
            layers.append(ResidualLinear(hidden_dim, hidden_dim))
            layers.append(nl())
        layers.append(
            ResidualLinear(hidden_dim, out_dim) if out_dim == hidden_dim else MyLinear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        flat = x.view(-1, x.shape[-1])
        ret_flat = self.main(flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        return ret


class ResidualLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResidualLinear, self).__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x):
        z = self.linear(x) + x
        return z


class MyLinear(nn.Linear):
    def forward(self, x):
        if x.dtype == torch.half:
            return half_linear(x, self.weight, self.bias)
        else:
            return single_linear(x, self.weight, self.bias)


def half_linear(x, weight, bias):
    return F.linear(x, weight.half(), bias.half())


def single_linear(x, weight, bias):
    return F.linear(x, weight, bias)
