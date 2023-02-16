from datetime import datetime as dt
import torch
import os
import sys
import numpy as np
import pickle
import collections

_verbose = False


def log(msg):
    print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S.%f'), msg))
    sys.stdout.flush()


def vlog(msg):
    if _verbose:
        print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
        sys.stdout.flush()


def flog(msg, outfile):
    msg = '{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(msg)
    sys.stdout.flush()
    try:
        with open(outfile, 'a') as f:
            f.write(msg + '\n')
    except Exception as e:
        log(e)


def load_pkl(pkl):
    with open(pkl, 'rb') as f:
        x = pickle.load(f)
    return x


def save_pkl(data, out_pkl, mode='wb'):
    if mode == 'wb' and os.path.exists(out_pkl):
        vlog(f'Warning: {out_pkl} already exists. Overwriting.')
    with open(out_pkl, mode) as f:
        pickle.dump(data, f)


def to_numpy(t):
    return t.detach().cpu().numpy()


def R_from_eman(a, b, y):
    a *= np.pi / 180.
    b *= np.pi / 180.
    y *= np.pi / 180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rb = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    Ry = np.array(([cy, -sy, 0], [sy, cy, 0], [0, 0, 1]))
    R = np.dot(np.dot(Ry, Rb), Ra)
    # handling EMAN convention mismatch for where the origin of an image is (bottom right vs top right)
    R[0, 1] *= -1
    R[1, 0] *= -1
    R[1, 2] *= -1
    R[2, 1] *= -1
    return R


def R_from_relion(a, b, y):
    a *= np.pi / 180.
    b *= np.pi / 180.
    y *= np.pi / 180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rb = np.array([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
    Ry = np.array(([cy, -sy, 0], [sy, cy, 0], [0, 0, 1]))
    R = np.dot(np.dot(Ry, Rb), Ra)
    R[0, 1] *= -1
    R[1, 0] *= -1
    R[1, 2] *= -1
    R[2, 1] *= -1
    return R


def R_from_relion_scipy(euler_, degrees=True):
    '''Nx3 array of RELION euler angles to rotation matrix'''
    from scipy.spatial.transform import Rotation as RR
    euler = euler_.copy()
    if euler.shape == (3,):
        euler = euler.reshape(1, 3)
    euler[:, 0] += 90
    euler[:, 2] -= 90
    f = np.ones((3, 3))
    f[0, 1] = -1
    f[1, 0] = -1
    f[1, 2] = -1
    f[2, 1] = -1
    rot = RR.from_euler('zxz', euler, degrees=degrees).as_matrix() * f
    return rot


def R_to_relion_scipy(rot, degrees=True):
    '''Nx3x3 rotation matrices to RELION euler angles'''
    from scipy.spatial.transform import Rotation as RR
    if rot.shape == (3, 3):
        rot = rot.reshape(1, 3, 3)
    assert len(rot.shape) == 3, "Input must have dim Nx3x3"
    f = np.ones((3, 3))
    f[0, 1] = -1
    f[1, 0] = -1
    f[1, 2] = -1
    f[2, 1] = -1
    euler = RR.from_matrix(rot * f).as_euler('zxz', degrees=True)
    euler[:, 0] -= 90
    euler[:, 2] += 90
    euler += 180
    euler %= 360
    euler -= 180
    if not degrees:
        euler *= np.pi / 180
    return euler


def xrot(tilt_deg):
    '''Return rotation matrix associated with rotation over the x-axis'''
    theta = tilt_deg * np.pi / 180
    tilt = np.array([[1., 0., 0.],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return tilt


def make_summary(pose_pred_params, w_eps, bimodal=False, N_img=100):
    # fixme: only implemented for bimodal mode for now
    if bimodal:
        pose_params_summary = {}
        pose_params_summary['R1'] = pose_pred_params['R1'][:N_img].detach().cpu().numpy()
        pose_params_summary['R2'] = pose_pred_params['R2'][:N_img].detach().cpu().numpy()
        w_eps = w_eps.reshape(2, -1, 3)
        pose_params_summary['w_eps_0'] = w_eps[0, :N_img].detach().cpu().numpy()
        pose_params_summary['w_eps_1'] = w_eps[1, :N_img].detach().cpu().numpy()
        pose_params_summary['p'] = pose_pred_params['p'][:N_img].detach().cpu().numpy()
    else:
        pose_params_summary = {}
    return pose_params_summary


def get_w_eps_std(w_eps, pose_pred_params, bimodal=False):
    if bimodal:
        p = pose_pred_params['p']
        p_cat = torch.cat((1. - p, p), 0)
        w_eps_pred = w_eps[p_cat.reshape(-1) > 0.5]
        w_eps_std = torch.std(w_eps_pred).item()
    else:
        w_eps_std = torch.std(w_eps).item()
    return w_eps_std
