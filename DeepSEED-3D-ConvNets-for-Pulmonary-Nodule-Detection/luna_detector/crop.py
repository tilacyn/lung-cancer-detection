import time
import numpy as np
import warnings
from scipy.ndimage import zoom

class Crop(object):
    def __init__(self, config, random=False):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
        self.random = random

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        """
        bboxes - array of 4 (3 coord and diameter)
        target - the original bbox (array of 4)
        isRand - should we take target into account when creating crop or not

        return:
        bboxes - array of bboxes, each bbox is an array of 4
        """
        # print('crop input bboxes %s' % bboxes)
        if self.random:
            np.random.seed(int(time.time() * 1000) % 100000)
        # print('crop input target %s' % target)
        if isScale:
            radiusLim = [8., 100.]
            scaleLim = [0.75, 1.25]
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
                , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        # print('isRand: {}'.format(isRand))
        # print('target (in crop call method): {}'.format(target))
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            #     randomized crops including target (or not including depending on isRand
            if s > e:
                start.append(int(np.random.randint(e, s)))  # !
                # print('start append {}'.format(start[-1]))
            else:
                start.append(int(target[i] - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2)))
            # print(s, e, bound_size)

        # print('start %s' % start)
        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        # print('normstart %s' % normstart)
        # print('normsize %s' % normsize)
        xx, yy, zz = np.meshgrid(
            np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] // self.stride),
            np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] // self.stride),
            np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] // self.stride),
            indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        pad = [[0, 0]]
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])
               ]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
        if isRand:
            target = np.array([np.nan, np.nan, np.nan, np.nan])
        return crop, target, bboxes, coord, 1
