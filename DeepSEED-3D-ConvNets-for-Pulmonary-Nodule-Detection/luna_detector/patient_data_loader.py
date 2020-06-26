from torch.utils.data import Dataset
from data_loader import get_filenames_and_labels
from label_mapping import LabelMapping
import time
import warnings
import os
import numpy as np


def pad(crop, size):
    s = crop.shape
    pad_sizes = [[size - s[0], 0], [size - s[1], 0], [size - s[2], 0]]
    # print(pad_sizes)
    return np.pad(crop, pad_sizes, 'minimum')


class PatientDataLoader(Dataset):
    def __init__(self, data_dir, split_path, config, split_comber=None, start=0, end=0):
        self.stride = config['stride']
        self.split_comber = split_comber
        self.crop_size = config['crop_size']
        idcs = split_path
        self.filenames, self.sample_bboxes = get_filenames_and_labels(idcs, start, end, data_dir)
        self.label_mapping = LabelMapping(config, 'train')
        self.cropper = Cropper(config)

    def __getitem__(self, idx):
        # bbox = self.bboxes[idx]
        filename = self.filenames[idx]
        imgs = np.load(filename)
        bboxes = self.sample_bboxes[idx]
        crops, labels = self.cropper.crop(imgs[0], bboxes)
        flatten_shape = np.concatenate([[-1], self.crop_size])
        crops = crops.reshape(flatten_shape)
        crops = crops[:, np.newaxis, ...]
        labels = labels.reshape(-1)
        xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, crops.shape[2] // self.stride),
                                 np.linspace(-0.5, 0.5, crops.shape[3] // self.stride),
                                 np.linspace(-0.5, 0.5, crops.shape[4] // self.stride), indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        crops = (crops.astype(np.float32) - 128) / 128
        return crops, labels, coord

    def __len__(self):
        return len(self.sample_bboxes)


class Cropper(object):
    def __init__(self, config):
        self.delta = 0
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
        self.apply3d = lambda f, arr: [[[f(crop) for crop in zs] for zs in ys] for ys in arr]
        self.apply3d_enumerated = lambda f, arr: [
            [[f(crop, i, j, k) for k, crop in enumerate(y)] for j, y in enumerate(yx)] for i, yx in enumerate(arr)]

    def crop(self, imgs, target):
        fit_times = [1 + imgs.shape[i] // 64 for i in range(3)]
        split_spaces = [np.int32(np.linspace(0, imgs.shape[i], fit_times[i] + 1)[1:-1]) for i in range(3)]
        split_spaces = np.array(split_spaces)
        crops = np.split(imgs, split_spaces[0])
        crops = [np.split(crop, split_spaces[1], axis=1) for crop in crops]
        crops = [[np.split(crop, split_spaces[2], axis=2) for crop in crop_line] for crop_line in crops]
        labels = self.get_labels(crops, target)
        crops = self.apply3d(pad, crops)
        return np.array(crops), np.array(labels)

    def get_labels(self, crops, target):
        def get_label(crop, i, j, k):
            sh = crop.shape
            left_coord = np.array([sh[0] * i, sh[1] * j, sh[2] * k])
            right_coord = np.array([sh[0] * (i + 1), sh[1] * (j + 1), sh[2] * (k + 1)])

            def has_label(bbox, delta):
                left_bbox_coord = bbox[:-1] - bbox[-1]
                right_bbox_coord = bbox[:-1] + bbox[-1]
                is_bbox_corner_inside = lambda corner: np.all(corner > left_coord + delta) and np.all(corner < right_coord - delta)
                return is_bbox_corner_inside(left_bbox_coord) or is_bbox_corner_inside(right_bbox_coord)

            for bbox in target:
                if has_label(bbox, self.delta):
                    return True
            return False

        labels = self.apply3d_enumerated(get_label, crops)
        return labels
