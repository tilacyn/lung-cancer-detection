import numpy as np
import torch
from torch.utils.data import Dataset
import os
import collections
import random
from layers_se import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import pdb


class DicomDdetector(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
        self.crop = Crop([64, 64, 64], imgs)
        self.crop.calc_start_array()

    def __getitem__(self, idx, split=None):
        
        sample, coord = self.crop(idx)

        sample = (sample.astype(np.float32) - 128) / 128
        return torch.from_numpy(sample), coord
        

    def __len__(self):
        return len(self.crop.start_array)





class Crop(object):
    def __init__(self, crop_size, imgs):
        self.imgs = imgs
        self.crop_size = crop_size
        self.start_array = []
       
    def calc_start_array(self):
        x_start = 0
        y_start = 0
        z_start = 0
        print("img", self.imgs.shape)
        while self.imgs.shape[1] - x_start > self.crop_size[0]:
            y_start = 0
            x_start += self.crop_size[0]
            if x_start > self.imgs.shape[1] - self.crop_size[0]:
                x_start = self.imgs.shape[1] - self.crop_size[0]
            while self.imgs.shape[2] - y_start > self.crop_size[1]:
                z_start = 0
                y_start += self.crop_size[1]
                if y_start > self.imgs.shape[2] - self.crop_size[1]:
                    y_start = self.imgs.shape[2] - self.crop_size[1]
                while self.imgs.shape[3] - z_start > self.crop_size[2]:
                    z_start += self.crop_size[2]
                    if z_start > self.imgs.shape[3] - self.crop_size[2]:
                        z_start = self.imgs.shape[3] - self.crop_size[2]
                        
                    self.start_array.append([x_start, y_start, z_start])
        print("start", self.start_array)
        
    def __call__(self, idx):
        print("idx", idx)
        #print("start array", self.start_array)
        #print("size", imgs.shape)
        
        
        #start = self.start_array[idx+10]#[240, 110, 240]
        start = [240, 110, 240]
        print("start", start)
        stride = 4

        # print('start %s' % start)
        normstart = np.array(start).astype('float32') / np.array(self.imgs.shape[1:]) - 0.5
        normsize = np.array(self.crop_size).astype('float32') / np.array(self.imgs.shape[1:])

        xx, yy, zz = np.meshgrid(
            np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] // stride),
            np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] // stride),
            np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] // stride),
            indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
            

        crop = self.imgs[:,
               max(start[0], 0):min(start[0] + self.crop_size[0], self.imgs.shape[1]),
               max(start[1], 0):min(start[1] + self.crop_size[1], self.imgs.shape[2]),
               max(start[2], 0):min(start[2] + self.crop_size[2], self.imgs.shape[3])
               ]
        print("crop", crop.shape)
   
        return crop, coord

