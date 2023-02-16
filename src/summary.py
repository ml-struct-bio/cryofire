"""
Summaries
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

import analysis
import fft
import metrics
import utils
import lie_tools

log = utils.log


def make_scalar_summary(writer, scalars, step):
    for key in scalars:
        writer.add_scalar(key, scalars[key], step)


def make_conf_summary(writer, conf, steps, colors_path, test=False, pca=None):
    if test:
        prefix = '(Test) '
    else:
        prefix = '(Train) '

    colors_gt = None
    if colors_path is not None:
        colors_gt = utils.load_pkl(colors_path)

    if conf.shape[1] > 1:
        if pca is None:
            pc, pca = analysis.run_pca(conf)
        else:
            pc = pca.transform(conf)

        fig = plt.figure(dpi=96, figsize=(5, 5))
        plt.scatter(pc[:, 0], pc[:, 1], color='k', s=2, alpha=.1)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        writer.add_figure(prefix + "PCA Embeddings", fig, global_step=steps)

        if colors_path is not None:
            # Colored PCA
            fig = plt.figure(dpi=96, figsize=(5, 5))
            n_conf = pc.shape[0]
            for i in range(n_conf):
                if i % 1000 < 100:  # only plot 10% of the points
                    plt.plot(pc[i, 0], pc[i, 1], 'o', alpha=.1, color=colors_gt[i], markersize=2)
            writer.add_figure(prefix + "PCA Colored Embeddings", fig, global_step=steps)

    if conf.shape[1] == 1:
        fig = plt.figure(dpi=96, figsize=(5, 5))
        plt.plot(conf[:, 0], 'k.')
        writer.add_figure(prefix + "1D plot", fig, global_step=steps)

    return pca


def make_img_summary(writer, in_dict, recon_y, output_mask, steps, encode_real=True, test=False):
    if test:
        prefix = '(Test) '
    else:
        prefix = '(Train) '

    batch_size = in_dict['y'].shape[0]

    # HT
    fig = plt.figure(dpi=96, figsize=(10, 4))
    # GT
    plt.subplot(121)
    y = utils.to_numpy(in_dict['y'][0])
    plt.imshow(y, cmap='plasma')
    plt.colorbar()
    plt.title('GT')
    # Pred
    plt.subplot(122)
    y_pred = torch.zeros_like(in_dict['y']).reshape(batch_size, -1)[0]
    y_pred[output_mask.binary_mask] = recon_y[0].to(dtype=y_pred.dtype)
    y_pred = utils.to_numpy(y_pred)
    resolution = int(np.sqrt(y_pred.shape[0]))
    y_pred = y_pred.reshape(resolution, resolution)
    plt.imshow(y_pred, cmap='plasma')
    plt.colorbar()
    plt.title('Prediction')
    writer.add_figure(prefix + "Hartley Transform", fig, global_step=steps)

    # HT (log)
    fig = plt.figure(dpi=96, figsize=(10, 4))
    # GT
    plt.subplot(121)
    plt.imshow(np.abs(y), norm=colors.LogNorm(), cmap='RdPu')
    plt.colorbar()
    plt.title('GT')
    # Pred
    plt.subplot(122)
    plt.imshow(np.abs(y_pred), norm=colors.LogNorm(), cmap='RdPu')
    plt.colorbar()
    plt.title('Prediction')
    writer.add_figure(prefix + "Log Hartley Transform", fig, global_step=steps)

    # Image
    fig = plt.figure(dpi=96, figsize=(10, 4))
    # GT
    plt.subplot(121)
    if encode_real:
        y_real = utils.to_numpy(in_dict['y_real'][0])
    else:
        y_real = fft.ihtn_center(y)
    plt.imshow(y_real)
    plt.colorbar()
    plt.title('GT')
    # Pred
    plt.subplot(122)
    y_real_pred = fft.ihtn_center(y_pred)
    plt.imshow(y_real_pred.reshape(resolution, resolution))
    plt.colorbar()
    plt.title('Prediction')
    writer.add_figure(prefix + "Image", fig, global_step=steps)


def make_pose_summary(writer, rots, trans, gt_dist_to_pix, path_gt, steps, ind, shift=False, test=False):
    if test:
        prefix = '(Test) '
    else:
        prefix = '(Train) '

    poses_gt = utils.load_pkl(path_gt)
    if poses_gt[0].ndim == 3:
        # contains translations
        rotmat_gt = torch.tensor(poses_gt[0]).float()
        trans_gt = torch.tensor(poses_gt[1]).float()
        if ind is not None:
            rotmat_gt = rotmat_gt[ind]
            trans_gt = trans_gt[ind]
    else:
        rotmat_gt = torch.tensor(poses_gt).float()
        trans_gt = None
        assert not shift, "Shift activated but trans not given in gt"
        if ind is not None:
            rotmat_gt = rotmat_gt[ind]

    rots = torch.tensor(rots).float()
    rot_pred_aligned, rot_gt_aligned, median_frob = metrics.align_rot_best(rotmat_gt, rots)
    rot_pred_aligned = torch.tensor(rot_pred_aligned).float()
    rot_gt_aligned = torch.tensor(rot_gt_aligned).float()
    writer.add_scalar(prefix + 'Median Frobenius Rotations', median_frob, steps)

    angles, mean_oop_error, median_oop_error = metrics.get_angular_error(
        rotmat_gt, rot_pred_aligned)
    writer.add_scalar(prefix + 'Mean Out-of-Plane Angular Error (deg)', mean_oop_error, steps)
    writer.add_scalar(prefix + 'Median Out-of-Plane Angular Error (deg)', median_oop_error, steps)

    fig = plt.figure(dpi=96, figsize=(7, 5))
    plt.hist(angles.flatten(), bins=100)
    plt.xlabel('Out-of-Plane Error (deg)')
    writer.add_figure(prefix + "Out-of-Plane Errors (deg)", fig, global_step=steps)

    euler_gt = lie_tools.rotmat_to_euler(rotmat_gt.numpy())
    euler_pred = lie_tools.rotmat_to_euler(torch.clone(rots).numpy())
    euler_pred_aligned = lie_tools.rotmat_to_euler(rot_pred_aligned.numpy())
    euler_gt_aligned = lie_tools.rotmat_to_euler(rot_gt_aligned.numpy())

    fig = plt.figure(dpi=96, figsize=(15, 5))
    plt.subplot(131)
    plt.plot(euler_gt[:, 0], euler_pred_aligned[:, 0], 'ro', alpha=.1)
    plt.xlabel('alpha gt')
    plt.ylabel('alpha pred')
    plt.subplot(132)
    plt.plot(euler_gt[:, 1], euler_pred_aligned[:, 1], 'go', alpha=.1)
    plt.xlabel('beta gt')
    plt.ylabel('beta pred')
    plt.subplot(133)
    plt.plot(euler_gt[:, 2], euler_pred_aligned[:, 2], 'bo', alpha=.1)
    plt.xlabel('gamma gt')
    plt.ylabel('gamma pred')
    writer.add_figure(prefix + "Euler Angles GT vs Pred (GT Frame)", fig, global_step=steps)

    fig = plt.figure(dpi=96, figsize=(15, 5))
    plt.subplot(131)
    plt.plot(euler_pred[:, 0], euler_gt_aligned[:, 0], 'ro', alpha=.1)
    plt.xlabel('alpha pred')
    plt.ylabel('alpha gt')
    plt.subplot(132)
    plt.plot(euler_pred[:, 1], euler_gt_aligned[:, 1], 'go', alpha=.1)
    plt.xlabel('beta pred')
    plt.ylabel('beta gt')
    plt.subplot(133)
    plt.plot(euler_pred[:, 2], euler_gt_aligned[:, 2], 'bo', alpha=.1)
    plt.xlabel('gamma pred')
    plt.ylabel('gamma gt')
    writer.add_figure(prefix + "Euler Angles GT vs Pred (Model Frame)", fig, global_step=steps)

    ip_angular_errors = np.min(np.concatenate([
        np.abs(euler_pred_aligned[:, 0] - euler_gt[:, 0])[None],
        np.abs(euler_pred_aligned[:, 0] - (euler_gt[:, 0] - 2.0 * np.pi))[None],
        np.abs(euler_pred_aligned[:, 0] - (euler_gt[:, 0] + 2.0 * np.pi))[None]
    ], 0), 0)
    ip_angular_errors = ip_angular_errors * 180.0 / np.pi
    mean_ip_error = np.mean(ip_angular_errors)
    median_ip_error = np.median(ip_angular_errors)
    writer.add_scalar(prefix + 'Mean In-Plane Angular Error (deg)', mean_ip_error, steps)
    writer.add_scalar(prefix + 'Median In-Plane Angular Error (deg)', median_ip_error, steps)

    fig = plt.figure(dpi=96, figsize=(7, 5))
    plt.hist(ip_angular_errors.flatten(), bins=100)
    plt.xlabel('In-Plane Error (deg)')
    writer.add_figure(prefix + "In-Plane Errors (deg)", fig, global_step=steps)

    unitvec_gt = torch.tensor([0, 0, 1], dtype=torch.float32).reshape(3, 1)
    # plot out-of-plane angles (not aligned)
    out_of_planes = torch.sum(rots * unitvec_gt, dim=-2)
    out_of_planes = out_of_planes.numpy()
    out_of_planes /= np.linalg.norm(out_of_planes, axis=-1, keepdims=True)
    azimuth, elevation = lie_tools.direction_to_azimuth_elevation(out_of_planes)
    fig = plt.figure(dpi=96, figsize=(10, 4))
    plt.subplot(111, projection="mollweide")
    plt.plot(azimuth.flatten(), elevation.flatten(), 'ro', alpha=0.1)
    plt.grid(True)
    writer.add_figure(prefix + "Predicted Out-of-Plane Angles (not aligned)", fig, global_step=steps)

    if shift:
        trans = torch.tensor(trans).float()
        trans_pred_corr, trans_gt_corr, mse, medse = metrics.get_trans_metrics(trans_gt, trans, rots, gt_dist_to_pix)
        writer.add_scalar(prefix + 'MSE Trans', mse, steps)
        writer.add_scalar(prefix + 'MedSE Trans', medse, steps)

        fig = plt.figure(dpi=96, figsize=(10, 4))
        plt.subplot(121)
        plt.plot(trans_gt_corr[:, 0], trans_pred_corr[:, 0], 'ko', alpha=.1)
        plt.xlabel('x gt')
        plt.subplot(122)
        plt.plot(trans_gt_corr[:, 1], trans_pred_corr[:, 1], 'ko', alpha=.1)
        plt.xlabel('y gt')
        writer.add_figure(prefix + "Trans GT vs Pred", fig, global_step=steps)
