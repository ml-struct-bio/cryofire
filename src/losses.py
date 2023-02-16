"""
Losses
"""

import torch

import utils

log = utils.log


def symmetric_loss(y_pred, y_gt, sym_loss_factor):
    """
    y_pred: [sym_loss_factor * batch_size, n_pts]
    y_gt: [sym_loss_factor * batch_size, n_pts]
    sym_loss_factor: int
    """
    n_pts = y_gt.shape[-1]
    y_pred = y_pred.reshape(sym_loss_factor, -1, n_pts)
    y_gt = y_gt.reshape(sym_loss_factor, -1, n_pts)
    distance_double = torch.mean((y_pred - y_gt) ** 2, -1)
    min_distances, activated_paths = torch.min(distance_double, 0)
    return min_distances.mean(), activated_paths


def kl_divergence_conf(latent_variables_dict):
    z_mu = latent_variables_dict['z']
    z_logvar = latent_variables_dict['z_logvar']
    kld = torch.mean(torch.sum((1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp()) / 2.0, dim=1), dim=0)
    return kld
