"""
Tools for dealing with SO(3) group and algebra
Adapted from https://github.com/pimdh/lie-vae
All functions are pytorch-ified
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def rotmat_to_euler(rotmat):
    """
    rotmat: [..., 3, 3] (numpy)
    output: [..., 3, 3]
    """
    return Rotation.from_matrix(rotmat.swapaxes(-2, -1)).as_euler('zxz')


def direction_to_azimuth_elevation(out_of_planes):
    """
    out_of_planes: [..., 3]
    up: Y
    plane: (Z, X)
    output: ([...], [...]) (azimuth, elevation)
    """
    elevation = np.arcsin(out_of_planes[..., 1])
    azimuth = np.arctan2(out_of_planes[..., 0], out_of_planes[..., 2])
    return azimuth, elevation


def s2s2_to_matrix(v1, v2=None):
    """
    Normalize 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix.
    """
    if v2 is None:
        assert v1.shape[-1] == 6
        v2 = v1[..., 3:]
        v1 = v1[..., 0:3]
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.cross(e1, e2)
    return torch.cat([e1[..., None, :], e2[..., None, :], e3[..., None, :]], -2)


def euler_to_rotmat(euler):
    """
    euler: [..., 3] (numpy)
    output: [..., 3, 3]
    """
    return Rotation.from_euler('zxz', euler).as_matrix().swapaxes(-2, -1)


def select_predicted_latent(pred_full, activated_paths):
    """
    rots_full: [sym_loss_factor * batch_size, ...]
    activated_paths: [batch_size]
    """
    batch_size = activated_paths.shape[0]
    pred_full = pred_full.reshape(-1, batch_size, *pred_full.shape[1:])
    list_arange = np.arange(batch_size)
    pred = pred_full[activated_paths, list_arange]
    return pred
