import sys
import argparse
import os
import json
import torch
import numpy as np
import pickle

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, main_dir)

from src import mrc
from src import utils
from src.lattice import Lattice
from src.models import CryoFIRE
from src.mask import CircularMask

log = utils.log


def dict_to_args(config_dict):
    args = argparse.ArgumentParser().parse_args([])
    for key, value in config_dict.items():
        setattr(args, key, value)
    if args.seed < 0:
        args.seed = np.random.randint(0, 10000)
    return args


def generate_volume(config_dict, weights, resolution, outdir, z_pkl=None):
    args = dict_to_args(config_dict)

    device = torch.device('cuda')
    resolution = resolution + 1 if resolution % 2 == 0 else resolution
    lattice = Lattice(resolution, extent=0.5, device=device)
    output_mask = CircularMask(lattice, lattice.D // 2)

    shared_cnn_params = {
        'depth_cnn': args.depth_cnn,
        'channels_cnn': args.channels_cnn,
        'kernel_size_cnn': args.kernel_size_cnn,
        'mask_type': args.input_mask
    }
    conf_regressor_params = {
        'z_dim': args.z_dim,
        'std_z_init': args.std_z_init,
        'variational': args.variational_het
    }
    hyper_volume_params = {
        'n_layers': args.hypervolume_layers,
        'hidden_dim': args.hypervolume_dim,
        'pe_type': args.pe_type,
        'pe_dim': args.pe_dim,
        'feat_sigma': args.feat_sigma,
        'domain': args.hypervolume_domain
    }

    model = CryoFIRE(
        lattice,
        output_mask,
        shared_cnn_params,
        conf_regressor_params,
        hyper_volume_params,
        no_trans=args.no_trans,
        sym_loss=args.sym_loss,
        sym_loss_factor=args.sym_loss_factor,
        use_gt_poses=args.use_gt_poses
    )
    model.to(device)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_norm = (0., 1.)

    z = None
    if z_pkl is not None:
        with open(z_pkl, 'rb') as f:
            z = pickle.load(f)

    if z is None:
        log('Generating volume...')
        out_mrc = '{}/volume.mrc'.format(outdir)
        vol = model.eval_volume(data_norm, zval=z)
        mrc.write(out_mrc, vol.astype(np.float32))
    else:
        n_vol = z.shape[0]
        for i in range(n_vol):
            log('Generating volume {}...'.format(i))
            out_mrc = '{}/volume.{}.mrc'.format(outdir, i)
            vol = model.eval_volume(data_norm, zval=z[i].reshape(-1))
            mrc.write(out_mrc, vol.astype(np.float32))

    log('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config filename')
    parser.add_argument('--weights', required=True, help='Weights of the model (.pkl)')
    parser.add_argument('--resolution', required=True, help='Resolution')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--z', default=None, help='Latent z to be sampled (.pkl)')
    arguments = parser.parse_args()

    relative_config_path = os.path.join('configs/', arguments.config + '.json')
    with open(os.path.join(main_dir, relative_config_path), 'r') as f:
        config = json.load(f)

    generate_volume(config, arguments.weights, arguments.resolution, arguments.out_dir, z_pkl=arguments.z)
