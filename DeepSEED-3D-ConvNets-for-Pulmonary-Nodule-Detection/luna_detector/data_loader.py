#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:53:10 2018

@author: ym
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
from scipy.ndimage.interpolation import rotate
from crop import Crop
from label_mapping import LabelMapping
from path import augmented_prp



def get_filenames_and_labels(idcs, start, end, data_dir, with_augmented=False):
    idcs = [idx for idx in idcs if os.path.exists(os.path.join(data_dir, '%s_label.npy' % idx))]

    print('len idcs ', len(idcs))
    idcs = idcs[start: len(idcs) - end]
    filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
    labels = []

    for idx in idcs:
        l = np.load(os.path.join(data_dir, '%s_label.npy' % idx))
        if np.all(l == 0):
            l = np.array([])
        labels.append(l)
    if with_augmented:
        augmented_filenames = [filename for filename in os.listdir(augmented_prp) if filename.endswith('clean.npy') and filename[:3] in idcs]
        augmented_idcs = [int(filename[:3]) for filename in augmented_filenames]
        print('augmented idcs number ', len(augmented_idcs))
        augmented_labels = [np.load(os.path.join(augmented_prp, '%s_label.npy' % idx)) for idx in augmented_idcs]
        augmented_filepaths = [os.path.join(augmented_prp, filename) for filename in augmented_filenames]
        filenames += augmented_filepaths
        labels += augmented_labels
        print('augmented labels')
        print(augmented_labels)
    print('len(labels) ', len(labels))


    sample_bboxes = labels
    return filenames, sample_bboxes


class LungNodule3Ddetector(Dataset):
    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None, start=0, end=0, r_rand=None, with_augmented=False):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']
        self.stride = config['stride']
        sizelim = config['sizelim'] / config['reso']
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop'] if r_rand is None else r_rand
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        idcs = split_path
        self.filenames, self.sample_bboxes = get_filenames_and_labels(idcs, start, end, data_dir, with_augmented)

        if self.phase != 'test':
            self.bboxes = []

            for i, l in enumerate(self.sample_bboxes):
                if len(l) > 0:
                    for t in l:
                        if t[3] > sizelim:
                            self.bboxes += [[np.concatenate([[i], t])]]
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[i], t])]] * 2
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[i], t])]] * 4
            self.bboxes = np.concatenate(self.bboxes, axis=0)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        is_random_img = False
        if idx >= len(self.bboxes):
            idx = idx % len(self.bboxes)
            is_random_img = np.random.randint(2)

        is_random = np.random.randint(2)

        if not is_random_img:
            bbox = self.bboxes[idx]
            filename = self.filenames[int(bbox[0])]
            imgs = np.load(filename)
            bboxes = self.sample_bboxes[int(bbox[0])]
            isScale = self.augtype['scale'] and (self.phase == 'train')
            sample, target, bboxes, coord, _ = self.crop(imgs, bbox[1:], bboxes, isScale, is_random)
            if self.phase == 'train' and not is_random:
                sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                        ifflip=self.augtype['flip'],
                                                        ifrotate=self.augtype['rotate'],
                                                        ifswap=self.augtype['swap'])
        else:
            randimid = np.random.randint(len(self.filenames))
            filename = self.filenames[randimid]
            imgs = np.load(filename)
            bboxes = self.sample_bboxes[randimid]
            isScale = self.augtype['scale'] and (self.phase == 'train')
            sample, target, bboxes, coord, _ = self.crop(imgs, [], bboxes, isScale=False, isRand=True)
        # print(target)
        label = self.label_mapping(sample.shape[1:], target, bboxes)
        sample = (sample.astype(np.float32) - 128) / 128
        # print('sample shape: ', sample.shape)
        return torch.from_numpy(sample), torch.from_numpy(label), coord


    def __len__(self):
        if self.phase == 'train':
            return int(len(self.bboxes) / (1 - self.r_rand))
        elif self.phase == 'val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = (np.random.rand() - 0.5) * 20
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord


def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
