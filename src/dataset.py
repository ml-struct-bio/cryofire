import multiprocessing as mp
import os
import sys
from multiprocessing import Pool
import numpy as np
import torch
from torch.utils import data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils
import starfile
import mrc
import fft


log = utils.log


def load_particles(mrcs_txt_star, lazy=False, datadir=None, relion31=False):
    """
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files,
    or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    """
    if mrcs_txt_star.endswith('.txt'):
        particles = mrc.parse_mrc_list(mrcs_txt_star, lazy=lazy)
    elif mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir,
                                                                                               lazy=lazy)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star)  # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir,
                                                                                                   lazy=lazy)
            else:
                raise RuntimeError(e)
    elif mrcs_txt_star.endswith('.cs'):
        particles = starfile.csparc_get_particles(mrcs_txt_star, datadir, lazy)
    else:
        particles, _ = mrc.parse_mrc(mrcs_txt_star, lazy=lazy)
    return particles


class LazyMRCData(data.Dataset):
    """
    Class representing an .mrcs stack file -- images loaded on the fly
    """

    def __init__(self, mrcfile, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None,
                 relion31=False, window_r=0.85, flog=None):
        log = flog if flog is not None else utils.log
        assert not keepreal, 'Not implemented error'
        particles = load_particles(mrcfile, True, datadir=datadir, relion31=relion31)
        if ind is not None:
            particles = [particles[x] for x in ind]
        n_particles = len(particles)
        ny, nx = particles[0].get().shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(n_particles, ny, nx))
        self.particles = particles.astype(np.float32)
        self.N = n_particles
        self.D = ny + 1  # after symmetrizing HT
        self.invert_data = invert_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        self.window = window_mask(ny, window_r, .99) if window else None

    def estimate_normalization(self, n=1000):
        n = min(n, self.N)
        imgs = np.asarray([fft.ht2_center(self.particles[i].get()) for i in range(0, self.N, self.N // n)])
        if self.invert_data:
            imgs *= -1
        imgs = fft.symmetrize_ht(imgs)
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        log('Normalizing HT by {} +/- {}'.format(*norm))
        return norm

    def get(self, i):
        img = self.particles[i].get()
        if self.window is not None:
            img *= self.window
        img = fft.ht2_center(img).astype(np.float32)
        if self.invert_data:
            img *= -1
        img = fft.symmetrize_ht(img)
        img = (img - self.norm[0]) / self.norm[1]
        return img

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.get(index), index


def window_mask(resolution, in_rad, out_rad):
    assert resolution % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, resolution, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, resolution, endpoint=False, dtype=np.float32))
    r = (x0 ** 2 + x1 ** 2) ** .5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r - in_rad) / (out_rad - in_rad)))
    return mask


class MRCData(data.Dataset):
    """
    Class representing an .mrcs stack file
    """

    def __init__(self, mrcfile, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None,
                 relion31=False, max_threads=16, window_r=0.7, flog=None, lazy=False, poses_gt_pkl=None):
        log = flog if flog is not None else utils.log
        self.lazy = lazy
        if ind is not None:
            particles_real = load_particles(mrcfile, lazy=True, datadir=datadir, relion31=relion31)
            log('Particles loaded')
            particles_real = np.array([particles_real[i].get() for i in ind])
            log('Particles filtered')
        else:
            particles_real = load_particles(mrcfile, lazy=lazy, datadir=datadir, relion31=relion31)

        if not lazy:
            n_particles, ny, nx = particles_real.shape
            assert ny == nx, "Images must be square"
            assert ny % 2 == 0, "Image size must be even"
            log('Loaded {} {}x{} images'.format(n_particles, ny, nx))

            # Real space window
            if window:
                log(f'Windowing images with radius {window_r}')
                particles_real *= window_mask(ny, window_r, .99)

            # compute HT
            log('Computing FFT')
            max_threads = min(max_threads, mp.cpu_count())
            if max_threads > 1:
                log(f'Spawning {max_threads} processes')
                with Pool(max_threads) as p:
                    particles = np.asarray(p.map(fft.ht2_center, particles_real), dtype=np.float32)
            else:
                particles = []
                for i, img in enumerate(particles_real):
                    if i % 1000 == 0:
                        log('{} FFT computed'.format(i))
                    particles.append(fft.ht2_center(img))
                particles = np.asarray(particles, dtype=np.float32)
                log('Converted to FFT')

            if invert_data:
                particles *= -1

            # symmetrize HT
            log('Symmetrizing image data')
            particles = fft.symmetrize_ht(particles)

            # normalize
            if norm is None:
                norm = [np.mean(particles), np.std(particles)]
                norm[0] = 0
            particles = (particles - norm[0]) / norm[1]
            log('Normalized HT by {} +/- {}'.format(*norm))

            self.particles = particles
            self.N = n_particles
            self.D = particles.shape[1]  # ny + 1 after symmetrizing HT
            self.norm = norm
            self.keepreal = keepreal
            if keepreal:
                # self.particles_real = particles_real
                imgs = particles_real.astype(np.float32)
                norm_real = [np.mean(imgs), np.std(imgs)]
                norm_real[0] = 0
                imgs = (imgs - norm_real[0]) / norm_real[1]
                log('Normalized real space images by {} +/- {}'.format(*norm_real))
                self.imgs = imgs
                self.norm_real = norm_real
        else:
            self.particles_real = particles_real

            particles_real_sample = np.array([particles_real[i].get() for i in range(1000)])
            n_particles, ny, nx = particles_real_sample.shape

            self.window = window
            self.window_r = window_r
            if window:
                log(f'Windowing images with radius {window_r}')
                particles_real_sample *= window_mask(ny, window_r, .99)

            max_threads = min(max_threads, mp.cpu_count())
            log(f'Spawning {max_threads} processes')
            with Pool(max_threads) as p:
                particles_sample = np.asarray(p.map(fft.ht2_center, particles_real_sample), dtype=np.float32)

            self.invert_data = invert_data
            if invert_data:
                particles_sample *= -1

            particles_sample = fft.symmetrize_ht(particles_sample)

            if norm is None:
                norm = [np.mean(particles_sample), np.std(particles_sample)]
                norm[0] = 0

            self.norm = norm
            self.D = particles_sample.shape[1]
            self.N = len(particles_real)
            self.keepreal = keepreal
            if keepreal:
                norm_real = [np.mean(particles_real_sample), np.std(particles_real_sample)]
                norm_real[0] = 0
                self.norm_real = norm_real

        self.poses_gt_pkl = poses_gt_pkl
        if poses_gt_pkl is not None:
            self.poses_gt = utils.load_pkl(poses_gt_pkl)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if self.lazy:
            particle_real = self.particles_real[index].get()

            if self.window:
                particle_real *= window_mask(particle_real.shape[-1], self.window_r, .99)

            particle = fft.ht2_center(particle_real)

            if self.invert_data:
                particle *= -1

            particle = fft.symmetrize_ht(particle)

            particle = (particle - self.norm[0]) / self.norm[1]

            in_dict = {'y': particle,
                       'index': index}
            if self.keepreal:
                particle_real = (particle_real - self.norm_real[0]) / self.norm_real[1]
                in_dict['y_real'] = particle_real
        else:
            in_dict = {'y': self.particles[index],
                       'index': index}
            if self.keepreal:
                in_dict['y_real'] = self.imgs[index]

        if self.poses_gt_pkl is not None:
            if self.poses_gt[0].ndim == 3:
                rotmat_gt = torch.tensor(self.poses_gt[0]).float()[index]
                trans_gt = torch.tensor(self.poses_gt[1]).float()[index]
                in_dict['R'] = rotmat_gt
                in_dict['t'] = trans_gt
            else:
                rotmat_gt = torch.tensor(self.poses_gt).float()[index]
                in_dict['R'] = rotmat_gt

        return in_dict

    def get(self, index):
        return self.particles[index]


class PreprocessedMRCData(data.Dataset):
    def __init__(self, mrcfile, norm=None, ind=None, flog=None):
        log = flog if flog is not None else utils.log
        particles = load_particles(mrcfile, False)
        if ind is not None:
            particles = particles[ind]
        log(f'Loaded {len(particles)} {particles.shape[1]}x{particles.shape[1]} images')
        if norm is None:
            norm = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0]) / norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))
        self.particles = particles
        self.N = len(particles)
        self.D = particles.shape[1]  # ny + 1 after symmetrizing HT
        self.norm = norm

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        in_dict = {'y': self.particles[index],
                   'index': index}
        return in_dict

    def get(self, index):
        return self.particles[index]


class TiltMRCData(data.Dataset):
    """
    Class representing an .mrcs tilt series pair
    """

    def __init__(self, mrcfile, mrcfile_tilt, norm=None, keepreal=False, invert_data=False, ind=None, window=True,
                 datadir=None, window_r=0.85, flog=None):
        log = flog if flog is not None else utils.log
        if ind is not None:
            particles_real = load_particles(mrcfile, True, datadir)
            particles_tilt_real = load_particles(mrcfile_tilt, True, datadir)
            particles_real = np.array([particles_real[i].get() for i in ind], dtype=np.float32)
            particles_tilt_real = np.array([particles_tilt_real[i].get() for i in ind], dtype=np.float32)
        else:
            particles_real = load_particles(mrcfile, False, datadir)
            particles_tilt_real = load_particles(mrcfile_tilt, False, datadir)

        n_particles, ny, nx = particles_real.shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(n_particles, ny, nx))
        assert particles_tilt_real.shape == (n_particles, ny, nx), "Tilt series pair must have same dimensions as untilted particles"
        log('Loaded {} {}x{} tilt pair images'.format(n_particles, ny, nx))

        # Real space window
        if window:
            m = window_mask(ny, window_r, .99)
            particles_real *= m
            particles_tilt_real *= m

            # compute HT
        particles = np.asarray([fft.ht2_center(img) for img in particles_real]).astype(np.float32)
        particles_tilt = np.asarray([fft.ht2_center(img) for img in particles_tilt_real]).astype(np.float32)
        if invert_data:
            particles *= -1
            particles_tilt *= -1

        # symmetrize HT
        particles = fft.symmetrize_ht(particles)
        particles_tilt = fft.symmetrize_ht(particles_tilt)

        # normalize
        if norm is None:
            norm = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0]) / norm[1]
        particles_tilt = (particles_tilt - norm[0]) / norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.particles_tilt = particles_tilt
        self.norm = norm
        self.N = n_particles
        self.D = particles.shape[1]
        self.keepreal = keepreal
        if keepreal:
            self.particles_real = particles_real
            self.particles_tilt_real = particles_tilt_real

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        in_dict = {'y': self.particles[index],
                   'y_tilt': self.particles_tilt[index],
                   'index': index}
        return in_dict

    def get(self, index):
        return self.particles[index], self.particles_tilt[index]
