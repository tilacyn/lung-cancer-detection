from procedures.attack_pipeline import scan_manipulator

from config import *  # user configurations
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as ktf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
from utils.equalizer import *
import pickle
import numpy as np
import time
import scipy.ndimage
from utils.dicom_utils import *
from utils.utils import *


class MhdScanManipulator(scan_manipulator):
    def __init__(self, model_inj_path):
        super().__init__(model_inj_path)

    def get_load_function(self):
        return load_scan

        # tamper loaded scan at given voxel (index) coordinate
        # coord: E.g. vox: slice_indx, y_indx, x_indx    world: -324.3, 23, -234
        # action: 'inject' or 'remove'

    def tamper(self, coord, isVox=True):
        action = 'inject'
        if self.scan is None:
            print('Cannot tamper: load a target scan first.')
            return
        if (action == 'inject') and (self.generator_inj is None):
            print('Cannot inject: no injection model loaded.')
            return
        if (action == 'remove') and (self.generator_rem is None):
            print('Cannot inject: no removal model loaded.')
            return

        if action == 'inject':
            print('===Injecting Evidence===')
        else:
            print('===Removing Evidence===')
        if not isVox:
            print('converting world to vox')
            print('world ', coord)
            coord = world2vox(coord, self.scan_spacing, self.scan_orientation, self.scan_origin)
            print('vox ', coord)

        ### Cut Location
        print("Cutting out target region")
        cube_shape = get_scaled_shape(config["cube_shape"], 1 / self.scan_spacing)
        clean_cube_unscaled = cutCube(self.scan, coord, cube_shape)
        clean_cube, resize_factor = scale_scan(clean_cube_unscaled, self.scan_spacing)
        # Store backup reference
        sdim = int(np.max(cube_shape) * 1.3)
        clean_cube_unscaled2 = cutCube(self.scan, coord, np.array([sdim, sdim, sdim]))  # for noise touch ups later

        ### Normalize/Equalize Location
        print("Normalizing sample")
        if action == 'inject':
            clean_cube_eq = self.eq_inj.equalize(clean_cube)
            clean_cube_norm = (clean_cube_eq - self.norm_inj[0]) / ((self.norm_inj[2] - self.norm_inj[1]))
        else:
            clean_cube_eq = self.eq_rem.equalize(clean_cube)
            clean_cube_norm = (clean_cube_eq - self.norm_rem[0]) / ((self.norm_rem[2] - self.norm_rem[1]))

        ########  Inject Cancer   ##########

        ### Inject/Remove evidence
        if action == 'inject':
            print("Injecting evidence")
        else:
            print("Removing evidence")

        x = np.copy(clean_cube_norm)
        x[self.m_zlims[0]:self.m_zlims[1], self.m_xlims[0]:self.m_xlims[1], self.m_ylims[0]:self.m_ylims[1]] = 0
        x = x.reshape((1, config['cube_shape'][0], config['cube_shape'][1], config['cube_shape'][2], 1))
        if action == 'inject':
            x_mal = self.generator_inj.predict([x])
        else:
            x_mal = self.generator_rem.predict([x])
        x_mal = x_mal.reshape(config['cube_shape'])

        ### De-Norm/De-equalize
        print("De-normalizing sample")
        thr = 0.3
        x_mal[x_mal > thr] = thr  # fix boundry overflow
        x_mal[x_mal < -thr] = -thr
        mal_cube_eq = x_mal * ((self.norm_inj[2] - self.norm_inj[1])) + self.norm_inj[0]
        mal_cube = self.eq_inj.dequalize(mal_cube_eq)
        # Correct for pixel norm error
        # fix overflow
        print('max mal cube ', np.max(mal_cube))
        bad = np.where(mal_cube > 2000)
        # mal_cube[bad] = np.median(clean_cube)
        for i in range(len(bad[0])):
            neiborhood = cutCube(mal_cube, np.array([bad[0][i], bad[1][i], bad[2][i]]),
                                 (np.ones(3) * 5).astype(int), -1000)
            mal_cube[bad[0][i], bad[1][i], bad[2][i]] = np.mean(neiborhood)
        # fix underflow
        mal_cube[mal_cube < -1000] = -1000

        ### Paste Location
        print("Pasting sample into scan")
        mal_cube_scaled, resize_factor = scale_scan(mal_cube, 1 / self.scan_spacing)
        self.scan = pasteCube(self.scan, mal_cube_scaled, coord)

        ### Noise Touch-ups
        print("Adding noise touch-ups...")
        noise_map_dim = clean_cube_unscaled2.shape
        ben_cube_ext = clean_cube_unscaled2
        mal_cube_ext = cutCube(self.scan, coord, noise_map_dim)
        local_sample = clean_cube_unscaled

        # Init Touch-ups
        if action == 'inject':  # inject type
            noisemap = np.random.randn(150, 200, 300) * np.std(local_sample[local_sample < -600]) * .6
            kernel_size = 3
            factors = sigmoid((mal_cube_ext + 700) / 70)
            k = kern01(mal_cube_ext.shape[0], kernel_size)
            for i in range(factors.shape[0]):
                factors[i, :, :] = factors[i, :, :] * k
        else:  # remove type
            noisemap = np.random.randn(150, 200, 200) * 30
            kernel_size = .1
            k = kern01(mal_cube_ext.shape[0], kernel_size)
            factors = None

        # Perform touch-ups
        if config[
            'copynoise']:  # copying similar noise from hard coded location over this lcoation (usually more realistic)
            benm = cutCube(self.scan, np.array(
                [int(self.scan.shape[0] / 2), int(self.scan.shape[1] * .43), int(self.scan.shape[2] * .27)]),
                           noise_map_dim)
            x = np.copy(benm)
            x[x > -800] = np.mean(x[x < -800])
            noise = x - np.mean(x)
        else:  # gaussian interpolated noise is used
            rf = np.ones((3,)) * (60 / np.std(local_sample[local_sample < -600])) * 1.3
            np.random.seed(np.int64(time.time()))
            noisemap_s = scipy.ndimage.interpolation.zoom(noisemap, rf, mode='nearest')
            noise = noisemap_s[:noise_map_dim, :noise_map_dim, :noise_map_dim]
        mal_cube_ext += noise

        if action == 'inject':  # Injection
            final_cube_s = np.maximum((mal_cube_ext * factors + ben_cube_ext * (1 - factors)), ben_cube_ext)
        else:  # Removal
            minv = np.min((np.min(mal_cube_ext), np.min(ben_cube_ext)))
            final_cube_s = (mal_cube_ext + minv) * k + (ben_cube_ext + minv) * (1 - k) - minv

        # make it not so bright
        # final_cube_s = self.reduce_brightness(self.scan, final_cube_s)
        self.scan = pasteCube(self.scan, final_cube_s, coord)
        print('touch-ups complete')

    def reduce_brightness(self, scan, cube):
        cube_max = np.max(cube)
        scan_max = np.max(scan)
        cube_min = np.min(cube)
        k = (scan_max - cube_min) / (cube_max - cube_min)
        k = 0.5
        cube = cube_min + (cube - cube_min) * k
        return cube
