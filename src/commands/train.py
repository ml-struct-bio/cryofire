import sys
import os
import argparse
import pickle
import json
import time
from datetime import datetime as dt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, main_dir)

from src import mrc
from src import utils
from src import dataset
from src import ctf
from src import summary
from src.lattice import Lattice
from src.losses import symmetric_loss, kl_divergence_conf
from src.models import CryoFIRE
from src.lie_tools import select_predicted_latent
from src.mask import CircularMask, FrequencyMarchingMask

log = utils.log
vlog = utils.vlog


def dict_to_args(config_dict):
    args = argparse.ArgumentParser().parse_args([])
    for key, value in config_dict.items():
        setattr(args, key, value)
    if args.seed < 0:
        args.seed = np.random.randint(0, 10000)
    return args


class Trainer:
    def __init__(self, config_dict):
        args = dict_to_args(config_dict)
        log(args)

        args.outdir = os.path.join(main_dir, args.outdir)
        # output directory
        if args.outdir is not None and not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        # load poses
        if args.pose is not None:
            assert os.path.exists(args.pose)

        # set the random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # set the device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        log('Use cuda {}'.format(self.use_cuda))

        # tensorboard writer
        self.summaries_dir = '{}/summaries'.format(args.outdir)
        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)
        self.writer = SummaryWriter(self.summaries_dir)
        log('Will write tensorboard summaries in {}'.format(self.summaries_dir))

        # load the particles
        if args.ind is not None:
            log('Filtering image dataset with {}'.format(args.ind))
            args.ind = pickle.load(open(args.ind, 'rb'))
            log("Done loading")
        self.data = dataset.MRCData(args.particles, norm=args.norm, keepreal=True, ind=args.ind,
                                    lazy=args.lazy, relion31=args.relion31, poses_gt_pkl=args.pose)
        self.n_img_dataset = self.data.N
        self.resolution = self.data.D
        # load the test set
        self.has_test_set = False
        if args.test_particles is not None:
            self.has_test_set = None
            self.data_test = dataset.MRCData(args.test_particles, norm=args.norm, keepreal=True,
                                             relion31=args.relion31, poses_gt_pkl=args.test_pose)
            self.n_img_test_dataset = self.data_test.N
            assert self.data_test.D == self.resolution

        # load ctf
        if args.ctf is not None:
            log('Loading ctf params from {}'.format(args.ctf))
            ctf_params = ctf.load_ctf_for_training(self.resolution - 1, args.ctf)
            if args.ind is not None:
                ctf_params = ctf_params[args.ind]
            assert ctf_params.shape == (self.n_img_dataset, 8)
            self.ctf_params = torch.tensor(ctf_params, device=self.device)
        else:
            self.ctf_params = None
        # load ctf test
        if args.test_ctf is not None:
            log('Loading test ctf params from {}'.format(args.test_ctf))
            ctf_params_test = ctf.load_ctf_for_training(self.resolution - 1, args.test_ctf)
            assert ctf_params_test.shape == (self.n_img_test_dataset, 8)
            self.ctf_params_test = torch.tensor(ctf_params_test, device=self.device)
        else:
            self.ctf_params_test = None

        # lattice
        self.lattice = Lattice(self.resolution, extent=0.5, device=self.device)

        # output mask
        if args.output_mask == 'circ':
            self.output_mask = CircularMask(self.lattice, self.lattice.D // 2)
        elif args.output_mask == 'frequency_marching':
            self.output_mask = FrequencyMarchingMask(
                self.lattice, self.lattice.D // 2, add_one_every=args.add_one_frequency_every)
        else:
            raise NotImplementedError

        # shared cnn
        shared_cnn_params = {
            'depth_cnn': args.depth_cnn,
            'channels_cnn': args.channels_cnn,
            'kernel_size_cnn': args.kernel_size_cnn,
            'mask_type': args.input_mask
        }
        # conformational encoder
        if args.z_dim > 0:
            log("Heterogeneous reconstruction with z_dim = {}".format(args.z_dim))
        else:
            log("Homogeneous reconstruction")
        conf_regressor_params = {
            'z_dim': args.z_dim,
            'std_z_init': args.std_z_init,
            'variational': args.variational_het
        }
        for key in conf_regressor_params.keys():
            log('{}: {}'.format(key, conf_regressor_params[key]))
        # hypervolume
        hyper_volume_params = {
            'n_layers': args.hypervolume_layers,
            'hidden_dim': args.hypervolume_dim,
            'pe_type': args.pe_type,
            'pe_dim': args.pe_dim,
            'feat_sigma': args.feat_sigma,
            'domain': args.hypervolume_domain
        }
        log("Initializing model...")
        self.model = CryoFIRE(
            self.lattice,
            self.output_mask,
            shared_cnn_params,
            conf_regressor_params,
            hyper_volume_params,
            no_trans=args.no_trans,
            sym_loss=args.sym_loss,
            sym_loss_factor=args.sym_loss_factor,
            use_gt_poses=args.use_gt_poses
        )
        log("Model initialized. Moving to GPU...")
        self.model.to(self.device)
        log(self.model)
        log('{} parameters in model'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        # initialization from a previous checkpoint
        if args.load:
            log('Loading checkpoint from {}'.format(args.load))
            checkpoint = torch.load(args.load)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            if 'output_mask_radius' in checkpoint:
                self.output_mask.update_radius(checkpoint['output_mask_radius'])
            self.model.train()
        else:
            self.start_epoch = 0

        # dataloaders
        self.data_generator = DataLoader(self.data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=True)
        if args.test_particles is not None:
            self.data_generator_test = DataLoader(self.data_test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers)

        # useful fields
        self.num_epochs = args.num_epochs
        self.pose_only_phase = args.pose_only_phase
        self.pose_only = True
        self.beta_conf = args.beta_conf
        self.loss_scale = args.loss_scale
        self.sym_loss = args.sym_loss
        self.sym_loss_factor = args.sym_loss_factor
        self.no_trans = args.no_trans
        self.log_interval = args.log_interval
        self.output_mask_type = args.output_mask
        self.verbose_time = args.verbose_time
        self.log_heavy_interval = args.log_heavy_interval
        self.colors = args.colors
        self.pose = args.pose
        self.z_dim = args.z_dim
        self.ind = args.ind
        self.test_colors = args.test_colors
        self.test_pose = args.test_pose
        self.outdir = args.outdir

        self.use_kl_divergence = True
        if args.z_dim == 0 or not args.variational_het or args.beta_conf < 1e-8:
            self.use_kl_divergence = False

        # placeholders for predicted latent variables
        self.predicted_rots = np.empty((self.n_img_dataset, 3, 3))
        self.predicted_trans = np.empty((self.n_img_dataset, 2)) if not args.no_trans else None
        self.predicted_act_paths = np.empty((self.n_img_dataset, 1)) if args.sym_loss else None
        self.predicted_rots_full = np.empty((self.sym_loss_factor, self.n_img_dataset, 3, 3)) if args.sym_loss else None
        self.predicted_conf = np.empty((self.n_img_dataset, args.z_dim)) if args.z_dim > 0 else None
        if self.has_test_set:
            self.predicted_rots_test = np.empty((self.n_img_test_dataset, 3, 3))
            self.predicted_trans_test = np.empty((self.n_img_test_dataset, 2)) if not self.no_trans else None
            self.predicted_conf_test = np.empty((self.n_img_test_dataset, self.z_dim)) if self.z_dim > 0 else None

        # placeholders for runtimes
        self.time_dataloading = []
        self.time_to_gpu = []
        self.time_ctf = []
        self.time_encoder = []
        self.time_decoder = []
        self.time_loss = []
        self.time_backward = []
        self.time_to_cpu = []

        # counters
        self.epoch = 0
        self.current_batch_images_count = 0
        self.total_batch_count = 0
        self.total_images_count = 0

    def train(self):
        log("--- Training Starts Now ---")
        t_0 = dt.now()

        self.predicted_rots = np.eye(3).reshape(1, 3, 3).repeat(self.n_img_dataset, axis=0)
        self.predicted_trans = np.zeros((self.n_img_dataset, 2)) if not self.no_trans else None
        self.predicted_act_paths = np.zeros((self.n_img_dataset, 1)) if self.sym_loss else None
        self.predicted_rots_full = np.eye(3).reshape(1, 1, 3, 3).repeat(
            self.sym_loss_factor, axis=0).repeat(self.n_img_dataset, axis=1) if self.sym_loss else None
        self.predicted_conf = np.zeros((self.n_img_dataset, self.z_dim)) if self.z_dim > 0 else None

        self.total_batch_count = 0
        self.total_images_count = 0
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            self.time_dataloading = []
            self.time_to_gpu = []
            self.time_ctf = []
            self.time_encoder = []
            self.time_decoder = []
            self.time_loss = []
            self.time_backward = []
            self.time_to_cpu = []

            self.epoch = epoch
            self.current_batch_images_count = 0

            end_time = time.time()
            in_dict = {}
            for in_dict in self.data_generator:
                self.train_step(in_dict, end_time=end_time)
                end_time = time.time()

            # image and pose summary
            if self.log_heavy_interval and epoch % self.log_heavy_interval == 0:
                self.make_full_summary(in_dict)
                self.save_latents()
                self.save_volume()
                self.save_model()

        t_total = dt.now() - t_0
        log('Finished in {} ({} per epoch)'.format(t_total, t_total / self.num_epochs))

    def train_step(self, in_dict, end_time):
        self.pose_only = False if self.total_images_count >= self.pose_only_phase else True

        self.time_dataloading.append(time.time() - end_time)

        y_gt = in_dict['y']
        ind = in_dict['index']
        self.total_batch_count += 1
        batch_size = len(y_gt)
        self.total_images_count += batch_size
        self.current_batch_images_count += batch_size

        # move to gpu
        start_time_gpu = time.time()
        for key in in_dict.keys():
            in_dict[key] = in_dict[key].to(self.device)
        self.time_to_gpu.append(time.time() - start_time_gpu)

        self.model.train()
        self.optim.zero_grad()

        # forward pass
        latent_variables_dict, y_pred, y_gt_processed = self.forward_pass(in_dict)

        # loss
        start_time_loss = time.time()
        total_loss, new_losses, activated_paths = self.loss(y_pred, y_gt_processed, latent_variables_dict)
        self.time_loss.append(time.time() - start_time_loss)

        # backward pass
        start_time_backward = time.time()
        total_loss.backward()
        self.optim.step()
        self.time_backward.append(time.time() - start_time_backward)

        # detach
        start_time_cpu = time.time()
        rot_pred, trans_pred, conf_pred, rot_pred_full, activated_paths = self.detach_latent_variables(
            latent_variables_dict, activated_paths)
        self.time_to_cpu.append(time.time() - start_time_cpu)
        # log
        if self.use_cuda:
            ind = ind.cpu()
        self.predicted_rots[ind] = rot_pred
        if not self.no_trans:
            self.predicted_trans[ind] = trans_pred
        if self.z_dim > 0:
            self.predicted_conf[ind] = conf_pred
        if self.sym_loss:
            self.predicted_act_paths[ind] = activated_paths.reshape(-1, 1)
            self.predicted_rots_full[:, ind, :, :] = rot_pred_full.reshape(-1, batch_size, 3, 3)

        # scalar summary
        if self.total_images_count % self.log_interval < batch_size:
            self.make_light_summary(total_loss, new_losses)

        # update output mask
        if hasattr(self.output_mask, 'update'):
            self.output_mask.update(self.total_images_count)

    def detach_latent_variables(self, latent_variables_dict, activated_paths):
        if self.sym_loss:
            rot_pred_full = latent_variables_dict['R'].detach().cpu().numpy()
            rot_pred = self.model.image_duplicator.compensate_transform_rotation(
                latent_variables_dict['R'].detach().cpu().numpy(),
                torch.clone(activated_paths).detach().cpu().numpy())
            trans_pred = self.model.image_duplicator.compensate_transform_translation(
                latent_variables_dict['t'], activated_paths).detach().cpu().numpy() if not self.no_trans else None
            conf_pred = select_predicted_latent(
                latent_variables_dict['z'], activated_paths).detach().cpu().numpy() if self.z_dim > 0 else None
            activated_paths = activated_paths.detach().cpu().numpy()
        else:
            rot_pred_full = None
            rot_pred = latent_variables_dict['R'].detach().cpu().numpy()
            trans_pred = latent_variables_dict['t'].detach().cpu().numpy() if not self.no_trans else None
            conf_pred = latent_variables_dict['z'].detach().cpu().numpy() if self.z_dim > 0 else None
            activated_paths = None
        return rot_pred, trans_pred, conf_pred, rot_pred_full, activated_paths

    def forward_pass(self, in_dict):
        ind = in_dict['index']
        batch_size = len(ind)

        # prepare CTFs
        ctf_params_local = self.ctf_params[ind] if self.ctf_params is not None else None
        start_time_ctf = time.time()
        if ctf_params_local is not None:
            freqs = self.lattice.freqs2d.unsqueeze(0).expand(
                batch_size, *self.lattice.freqs2d.shape) / ctf_params_local[:, 0].view(batch_size, 1, 1)
            ctf_local = ctf.compute_ctf(freqs, *torch.split(ctf_params_local[:, 1:], 1, 1)).view(batch_size,
                                                                                                 self.resolution,
                                                                                                 self.resolution)
        else:
            ctf_local = None
        self.time_ctf.append(time.time() - start_time_ctf)

        # forward pass
        start_time_encoder = time.time()
        latent_variables_dict = self.model.encode(in_dict, self.pose_only)
        self.time_encoder.append(time.time() - start_time_encoder)
        start_time_decoder = time.time()
        y_pred, y_gt_processed = self.model.decode(latent_variables_dict, ctf_local, in_dict['y'])
        self.time_decoder.append(time.time() - start_time_decoder)
        return latent_variables_dict, y_pred, y_gt_processed

    def loss(self, y_pred, y_gt, latent_variables_dict):
        """
        y_pred: [(sym_loss_factor * ) batch_size, n_pts]
        y_gt: [(sym_loss_factor * ) batch_size, D, D]
        """
        new_losses = {}

        # data loss
        if self.sym_loss:
            data_loss, activated_paths = symmetric_loss(y_pred, y_gt, self.sym_loss_factor)
        else:
            data_loss = F.mse_loss(y_pred, y_gt)
            activated_paths = None
        new_losses['Data Loss'] = data_loss.item()

        # KL divergence
        if not self.use_kl_divergence:
            total_loss = data_loss
        else:
            kld_conf = kl_divergence_conf(latent_variables_dict)
            total_loss = data_loss + self.beta_conf * kld_conf / self.resolution ** 2
            new_losses['KL Div. Conf.'] = kld_conf.item()

        return total_loss, new_losses, activated_paths

    def test(self):
        with torch.no_grad():
            for in_dict in self.data_generator_test:
                ind = in_dict['index']

                # move to gpu
                for key in in_dict.keys():
                    in_dict[key] = in_dict[key].to(self.device)

                latent_variables_dict, y_pred, y_gt_processed = self.forward_pass(in_dict)

                total_loss, new_losses, activated_paths = self.loss(y_pred, y_gt_processed, latent_variables_dict)

                # detach
                rot_pred, trans_pred, conf_pred, rot_pred_full, activated_paths = self.detach_latent_variables(
                    latent_variables_dict, activated_paths)
                # log
                if self.use_cuda:
                    ind = ind.cpu()
                self.predicted_rots_test[ind] = rot_pred
                if not self.no_trans:
                    self.predicted_trans_test[ind] = trans_pred
                if self.z_dim > 0:
                    self.predicted_conf_test[ind] = conf_pred

    def make_full_summary(self, in_dict):
        with torch.no_grad():
            for key in in_dict.keys():
                in_dict[key] = in_dict[key].to(self.device)
            latent_variables_dict, y_pred, y_gt_processed = self.forward_pass(in_dict)
        pca = self.make_heavy_summary(in_dict, y_pred, self.predicted_conf, self.predicted_rots,
                                      self.predicted_trans,
                                      self.colors, self.pose, self.ind)
        if self.has_test_set:
            self.predicted_rots_test = np.empty((self.n_img_test_dataset, 3, 3))
            self.predicted_trans_test = np.empty((self.n_img_test_dataset, 2)) if not self.no_trans else None
            self.predicted_conf_test = np.empty((self.n_img_test_dataset, self.z_dim)) if self.z_dim > 0 else None

            self.test()

            _ = self.make_heavy_summary(in_dict, y_pred, self.predicted_conf_test,
                                        self.predicted_rots_test, self.predicted_trans_test,
                                        self.test_colors, self.test_pose, indices=None, pca_previous=pca, test=True)

    def make_heavy_summary(self, in_dict, y_pred, predicted_conf, predicted_rots, predicted_trans, colors,
                           pose, indices, pca_previous=None, test=False):
        summary.make_img_summary(self.writer, in_dict, y_pred, self.output_mask, self.epoch + 1)
        # conformation
        pca = None
        if self.z_dim > 0:
            pca = summary.make_conf_summary(self.writer, predicted_conf, self.epoch + 1, colors, pca=pca_previous,
                                            test=test)
        # pose
        if self.pose:
            if not self.no_trans:
                gt_dist_to_pix = self.resolution
            else:
                gt_dist_to_pix = None
            summary.make_pose_summary(self.writer, predicted_rots, predicted_trans, gt_dist_to_pix,
                                      pose, self.epoch + 1, indices,
                                      shift=(not self.no_trans), test=test)
        return pca

    def make_light_summary(self, total_loss, new_losses):
        data_loss = new_losses['Data Loss']
        _total_loss = total_loss.item()
        log('# [Train Epoch: {}/{}] [{}/{} images] data loss={:.4f}, loss={:.4f}'.format(
            self.epoch + 1, self.num_epochs, self.current_batch_images_count, self.n_img_dataset,
            data_loss, _total_loss))

        scalars = new_losses
        if hasattr(self.output_mask, 'current_radius'):
            scalars['Radius Mask'] = self.output_mask.current_radius
        summary.make_scalar_summary(self.writer, scalars, self.total_images_count)

        if self.verbose_time:
            log('Dataloading Time {}'.format(np.mean(np.array(self.time_dataloading))))
            log('CPU -> GPU Time {}'.format(np.mean(np.array(self.time_to_gpu))))
            log('CTF Time {}'.format(np.mean(np.array(self.time_ctf))))
            log('Encoder Time {}'.format(np.mean(np.array(self.time_encoder))))
            log('Decoder Time {}'.format(np.mean(np.array(self.time_decoder))))
            log('Loss Time {}'.format(np.mean(np.array(self.time_loss))))
            log('Backward Time {}'.format(np.mean(np.array(self.time_backward))))
            log('GPU -> CPU Time {}'.format(np.mean(np.array(self.time_to_cpu))))

    def save_latents(self):
        out_pose = '{}/pose.{}.pkl'.format(self.outdir, self.epoch)
        if self.no_trans:
            with open(out_pose, 'wb') as f:
                pickle.dump(self.predicted_rots, f)
        else:
            with open(out_pose, 'wb') as f:
                pickle.dump((self.predicted_rots, self.predicted_trans), f)
        if self.z_dim > 0:
            out_conf = '{}/conf.{}.pkl'.format(self.outdir, self.epoch)
            with open(out_conf, 'wb') as f:
                pickle.dump(self.predicted_conf, f)
        if self.sym_loss:
            out_act_paths = '{}/act_paths.{}.pkl'.format(self.outdir, self.epoch)
            with open(out_act_paths, 'wb') as f:
                pickle.dump(self.predicted_act_paths, f)
            out_pose_full = '{}/pose_full.{}.pkl'.format(self.outdir, self.epoch)
            with open(out_pose_full, 'wb') as f:
                pickle.dump(self.predicted_rots_full, f)

        if self.has_test_set:
            out_pose_test = '{}/pose_test.{}.pkl'.format(self.outdir, self.epoch)
            if self.no_trans:
                with open(out_pose_test, 'wb') as f:
                    pickle.dump(self.predicted_rots_test, f)
            else:
                with open(out_pose_test, 'wb') as f:
                    pickle.dump((self.predicted_rots_test, self.predicted_trans_test), f)
            if self.z_dim > 0:
                out_conf_test = '{}/conf_test.{}.pkl'.format(self.outdir, self.epoch)
                with open(out_conf_test, 'wb') as f:
                    pickle.dump(self.predicted_conf_test, f)

    def save_volume(self):
        out_mrc = '{}/reconstruct.{}.mrc'.format(self.outdir, self.epoch)
        self.model.eval()
        if self.z_dim > 0:
            zval = self.predicted_conf[0].reshape(-1, 1)
        else:
            zval = None
        vol = self.model.eval_volume(self.data.norm, zval=zval)
        mrc.write(out_mrc, vol.astype(np.float32))

    def save_model(self):
        out_weights = '{}/weights.{}.pkl'.format(self.outdir, self.epoch)
        saved_objects = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }
        if hasattr(self.output_mask, 'current_radius'):
            saved_objects['output_mask_radius'] = self.output_mask.current_radius
        torch.save(saved_objects, out_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config filename')
    relative_config_path = os.path.join('configs/', parser.parse_args().config + '.json')
    with open(os.path.join(main_dir, relative_config_path), 'r') as f:
        config = json.load(f)
    utils._verbose = False
    trainer = Trainer(config)
    trainer.train()
