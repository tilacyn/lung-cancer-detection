import glob
import json
import os
import time
from os.path import join as opjoin

import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import pydicom as dicom
import numpy as np
from data_loader import LabelMapping
from data_loader import Crop
from numpy import array as na
import cv2

from config_training import config as config_training

from PIL import Image, ImageDraw

from matplotlib import pyplot as plt


def parseXML(scan_path):
    '''
    parse xml file
    args:
    xml file path
    output:
    nodule list
    [{nodule_id, roi:[{z, sop_uid, xy:[[x1,y1],[x2,y2],...]}]}]
    '''
    file_list = os.listdir(scan_path)
    xml_file = None
    for file in file_list:
        if '.' in file and file.split('.')[1] == 'xml':
            xml_file = file
            break
    prefix = "{http://www.nih.gov}"
    if xml_file is None:
        print('SCAN PATH: {}'.format(scan_path))
    tree = ET.parse(scan_path + '/' + xml_file)
    root = tree.getroot()
    readingSession_list = root.findall(prefix + "readingSession")
    nodules = []

    for session in readingSession_list:
        # print(session)
        unblinded_list = session.findall(prefix + "unblindedReadNodule")
        for unblinded in unblinded_list:
            nodule_id = unblinded.find(prefix + "noduleID").text
            edgeMap_num = len(unblinded.findall(prefix + "roi/" + prefix + "edgeMap"))
            if edgeMap_num >= 1:
                # it's segmentation label
                nodule_info = {}
                nodule_info['nodule_id'] = nodule_id
                nodule_info['roi'] = []
                roi_list = unblinded.findall(prefix + "roi")
                for roi in roi_list:
                    roi_info = {}
                    # roi_info['z'] = float(roi.find(prefix + "imageZposition").text)
                    roi_info['sop_uid'] = roi.find(prefix + "imageSOP_UID").text
                    roi_info['xy'] = []
                    edgeMap_list = roi.findall(prefix + "edgeMap")
                    for edgeMap in edgeMap_list:
                        x = float(edgeMap.find(prefix + "xCoord").text)
                        y = float(edgeMap.find(prefix + "yCoord").text)
                        xy = [x, y]
                        roi_info['xy'].append(xy)
                    nodule_info['roi'].append(roi_info)
                nodules.append(nodule_info)
    return nodules


def make_mask(image, image_id, nodules):
    height, width = image.shape
    # print(image.shape)
    filled_mask = np.full((height, width), 0, np.uint8)
    contoured_mask = np.full((height, width), 0, np.uint8)
    # todo OR for all masks
    for nodule in nodules:
        for roi in nodule['roi']:
            if roi['sop_uid'] == image_id:
                edge_map = roi['xy']
                cv2.fillPoly(filled_mask, np.int32([np.array(edge_map)]), 255)
                # cv2.polylines(contoured_mask, np.int32([np.array(edge_map)]), color=255, isClosed=False)

    # mask = np.swapaxes(np.array([contoured_mask, filled_mask]), 0, 2)
    # cv2.imwrite('kek0.jpg', image)
    # cv2.imwrite('kek1.jpg', filled_mask)
    return np.reshape(filled_mask, (height, width, 1))


def create_map_from_nodules(nodules):
    id2roi = {}
    for nodule in nodules:
        for roi in nodule['roi']:
            id = roi['sop_uid']
            if id not in id2roi:
                id2roi[id] = []
            id2roi[id].append(roi['xy'])
    return id2roi


def make_mask_for_rgb(image, image_id, nodules):
    filled_mask = image
    for nodule in nodules:
        for roi in nodule['roi']:
            if roi['sop_uid'] == image_id:
                edge_map = roi['xy']
                cv2.fillPoly(filled_mask, np.int32([np.array(edge_map)]), (255, 0, 0))
    return filled_mask


def imread(image_path):
    ds = dicom.dcmread(image_path)
    img = ds.pixel_array
    img_2d = img.astype(float)
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
    img_2d_scaled = np.uint8(img_2d_scaled)
    image = img_2d_scaled
    return image, ds


def resolve_bbox(dcms, id2roi):
    nodule_coordinates = []
    for i, dcm in enumerate(dcms):
        image, dcm_data = dcm
        if dcm_data.SOPInstanceUID not in id2roi:
            continue
        rois = id2roi[dcm_data.SOPInstanceUID]
        roi = rois[0]
        mean = np.mean(roi, axis=0)
        nodule_coordinates.append([i, mean[0], mean[1]])
    if len(nodule_coordinates) == 0:
        print('Nodule coordinates empty list')
        raise ValueError('invalid bbox')
    return np.concatenate((np.mean(nodule_coordinates, axis=0), [5.0]))


NPY_LOAD_MARKER = -5


class LIDCDataset(Dataset):
    def __init__(self, data_path, config, stard_idx, end_idx, load=False, isRand=False, phase='train', random=False):
        self.data_path = data_path
        self.ids = []
        self.start_idx = stard_idx
        self.end_idx = end_idx
        self.create_ids()
        self.phase = phase
        self.label_mapping = LabelMapping(config, 'train')
        self.crop = Crop(config, random)
        self.load = load
        self.lidc_npy_path = config_training['lidc-npy']
        self.isRand = isRand
        self.random = random

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        if self.phase == 'test':
            isRand = np.random.randint(0, 2) == 0
        else:
            isRand = self.isRand
        imgs, bbox = self.get_data_from_npy(idx)
        # print(bbox)
        sample, target, bboxes, coord, real_target = self.crop(imgs, bbox, [bbox], isScale=False, isRand=isRand)
        label = self.label_mapping(sample.shape[1:], target, bboxes)
        sample = (sample.astype(np.float32) - 128) / 128

        return torch.from_numpy(sample), \
               torch.from_numpy(label), \
               coord, \
               real_target

    def get_data_from_dcm(self, idx):
        dcms = []
        parent_path = self.ids[idx]
        for file in os.listdir(parent_path):
            if not file.endswith('dcm'):
                continue
            image, dcm_data = imread(opjoin(parent_path, file))
            if not has_slice_location(dcm_data):
                continue
            dcms.append((image, dcm_data))
        dcms.sort(key=lambda dcm: dcm[1].SliceLocation)
        nodules = parseXML(parent_path)
        id2roi = create_map_from_nodules(nodules)
        imgs = na([dcm[0] for dcm in dcms])
        imgs = imgs[np.newaxis, :]
        bbox = resolve_bbox(na(dcms), id2roi)
        return imgs, bbox

    def get_data_from_npy(self, idx):
        load_imgs_path = opjoin(self.lidc_npy_path, 'imgs_%d.npy' % idx)
        load_bbox_path = opjoin(self.lidc_npy_path, 'bbox_%d.npy' % idx)
        imgs = np.load(load_imgs_path)
        bbox = np.load(load_bbox_path)
        return imgs, bbox

    def get_preprocessed_data_from_npy(self, idx):
        make_load_path = lambda x: opjoin(self.lidc_npy_path, '%s_%d.npy' % (x, idx))
        sample = np.load(make_load_path('sample'))
        label = np.load(make_load_path('label'))
        coord = np.load(make_load_path('coord'))
        return sample, label, coord

    def save_npy(self, start, end):
        for i in range(start, end):
            print('processing %d' % i)
            try:
                imgs, bbox = self.get_data_from_dcm(i)
            except:
                imgs, bbox = self.get_data_from_dcm(i - 1)

            save_imgs_path = opjoin(self.lidc_npy_path, 'imgs_%d.npy' % i)
            save_bbox_path = opjoin(self.lidc_npy_path, 'bbox_%d.npy' % i)

            np.save(save_bbox_path, bbox)
            np.save(save_imgs_path, imgs)

    def __len__(self):
        return len(self.ids)

    def create_ids(self):
        with open('index.json', 'r') as read_file:
            self.ids = json.load(read_file)
            self.ids = self.ids[self.start_idx:self.end_idx]


def has_slice_location(dcm_data):
    try:
        slice_location = dcm_data.SliceLocation
        return True
    except:
        # print('No Slice Location')
        return False


def create_index(data_path):
    ids = []
    for root, _, files in os.walk(data_path):
        if glob.glob(opjoin(data_path, root, '*xml')):
            nodules = parseXML(opjoin(data_path, root))
            id2roi = create_map_from_nodules(nodules)
            if len(id2roi) == 0:
                continue
            ids.append(root)
    with open('index.json', 'w') as write_file:
        json.dump(ids, write_file)
