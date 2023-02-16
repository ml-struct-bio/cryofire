"""
Metrics
"""

import numpy as np
import torch

import utils


def get_ref_matrix(r1, r2, i, flip=False):
    if flip:
        return np.matmul(r2[i].T, _flip(r1[i]))
    else:
        return np.matmul(r2[i].T, r1[i])


def _flip(rot):
    x = np.diag([1, 1, -1]).astype(rot.dtype)
    return np.matmul(x, rot)


def align_rot(r1, r2, i, flip=False):
    if flip:
        return np.matmul(_flip(r2), get_ref_matrix(r1, r2, i, flip=True))
    else:
        return np.matmul(r2, get_ref_matrix(r1, r2, i, flip=False))


def align_rot_best(rot_gt_tensor, rot_pred_tensor, n_tries=100):
    """
    rot_gt_tensor: [n_rots, 3, 3]
    rot_pred_tensor: [n_rots, 3, 3]
    n_tries: int
    output: [n_rots, 3, 3] (numpy), float
    """
    rot_gt = rot_gt_tensor.clone().numpy()
    rot_pred = rot_pred_tensor.clone().numpy()

    median = []
    for i in range(n_tries):
        rot_pred_aligned = align_rot(rot_gt, rot_pred, i, flip=False)
        dists = frob_norm(rot_gt, rot_pred_aligned)
        median.append(np.median(dists))

    median_flip = []
    for i in range(n_tries):
        rot_pred_aligned = align_rot(rot_gt, rot_pred, i, flip=True)
        dists = frob_norm(rot_gt, rot_pred_aligned)
        median_flip.append(np.median(dists))

    if np.min(median) < np.min(median_flip):
        utils.log("Correct Handedness")
        i_best = np.argmin(median)
        alignment_matrix = get_ref_matrix(rot_gt, rot_pred, i_best, flip=False)
        rot_pred_aligned = np.matmul(rot_pred, alignment_matrix)
        rot_gt_aligned = np.matmul(rot_gt, alignment_matrix.T)
        median_frob = np.min(median)
    else:
        utils.log("Flipped Handedness")
        i_best = np.argmin(median_flip)
        alignment_matrix = get_ref_matrix(rot_gt, rot_pred, i_best, flip=True)
        rot_pred_aligned = np.matmul(_flip(rot_pred), alignment_matrix)
        rot_gt_aligned = _flip(np.matmul(rot_gt, alignment_matrix.T))
        median_frob = np.min(median_flip)

    return rot_pred_aligned, rot_gt_aligned, median_frob


def frob_norm(r1, r2):
    """
    r1: [n_rots, 3, 3]
    r2: [n_rots, 3, 3]
    output: float
    """
    return np.sum((r1 - r2) ** 2, axis=(1, 2))


def get_angular_error(rot_gt, rot_pred):
    """
    rot_gt: [n_rots, 3, 3]
    rot_pred: [n_rots, 3, 3]

    output: [n_rots] (numpy), float, float
    """
    unitvec_gt = torch.tensor([0, 0, 1], dtype=torch.float32).reshape(3, 1)

    out_of_planes_gt = torch.sum(rot_gt * unitvec_gt, dim=-2)
    out_of_planes_gt = out_of_planes_gt.numpy()
    out_of_planes_gt /= np.linalg.norm(out_of_planes_gt, axis=-1, keepdims=True)

    out_of_planes_pred = torch.sum(rot_pred * unitvec_gt, dim=-2)
    out_of_planes_pred = out_of_planes_pred.numpy()
    out_of_planes_pred /= np.linalg.norm(out_of_planes_pred, axis=-1, keepdims=True)

    angles = np.arccos(np.clip(np.sum(out_of_planes_gt * out_of_planes_pred, -1), -1.0, 1.0)) * 180.0 / np.pi

    return angles, np.mean(angles), np.median(angles)


def get_trans_metrics(trans_gt, trans_pred, rotmat, gt_dist_to_pix, correct_global_trans=False):
    # trans_pred in pixels
    # trans_gt in fraction
    trans_gt_corr = trans_gt * gt_dist_to_pix

    if correct_global_trans:
        b = torch.cat([(trans_pred - trans_gt_corr)[:, 0], (trans_pred - trans_gt_corr)[:, 1]], 0)
        matrix_a = torch.cat([rotmat[:, 0, :], rotmat[:, 1, :]], 0)
        u = torch.tensor(np.linalg.lstsq(matrix_a, b)[0])
        matrix_n = torch.tensor([[1, 0, 0], [0, 1, 0]]).reshape(1, 2, 3).float()
        batch_size = rotmat.shape[0]
        trans_pred_corr = trans_pred - torch.bmm(matrix_n.repeat(batch_size, 1, 1),
                                                 (u @ rotmat.permute(0, 2, 1)).reshape(-1, 3, 1)).reshape(-1, 2)
    else:
        trans_pred_corr = trans_pred

    dist = np.sum((trans_gt_corr.numpy() - trans_pred_corr.numpy()) ** 2, axis=1)

    mse = np.mean(dist)
    medse = np.median(dist)

    return trans_pred_corr, trans_gt_corr, mse, medse
