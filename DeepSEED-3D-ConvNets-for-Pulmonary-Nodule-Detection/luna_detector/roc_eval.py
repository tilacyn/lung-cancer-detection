import os
from importlib import import_module
from os.path import join as opjoin

import numpy as np
import torch
from tqdm import tqdm
from data_loader import LungNodule3Ddetector
from layers_se import *
from patient_data_loader import PatientDataLoader
from torch.utils.data import DataLoader
from abc import abstractmethod, ABC

from config_training import config as config_training

default_data_path = os.path.join('/content/drive/My Drive/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection',
                                   config_training['preprocess_result_path'])
base_path = '/content/drive/My Drive/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection'
luna_path = opjoin(base_path, 'luna_detector')

model = import_module('res18_se')


def run_test(ltest, left=-3.5, right=5, thr_number=20, mode='roc', net_number=0):
  result = {}
  f = ltest.test_luna if mode == 'roc' else ltest.froc_eval
  for thr in np.linspace(left, right, thr_number):
    result[thr] = f(thr, net_number)
  return result


class AbstractTest:
    def __init__(self, data_path=None, paths2model=None, start=0, end=0, r_rand=0.9, stage=0, all_tta=False):
        if paths2model is None:
            paths2model = ['']
        self.data_path = default_data_path if data_path is None else data_path
        self.stage = stage
        print('creating model')
        self.nets = [self._init_net(path2model) for path2model in paths2model]
        self.gp = GetPBB(self.config)
        self.start = start
        self.end = end
        self.r_rand = r_rand
        if all_tta:
            self.config['augtype'] = {'flip': True, 'swap': True, 'scale': True, 'rotate': True}
            print('ALL TTA')
        dataset = self.create_dataset()
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)

        self.outputs, self.targets = self.predict_on_data(data_loader)

    def _init_net(self, path2model):
        config, net, loss, get_pbb = model.get_model()
        net = net.cuda()
        checkpoint = torch.load(opjoin(luna_path, 'test_results', path2model))
        net.load_state_dict(checkpoint['state_dict'])
        self.config = config
        return net


    @abstractmethod
    def create_dataset(self):
        pass

    @abstractmethod
    def predict_on_data(self, data_loader):
        pass

    def transform_target(self, target):
        return target.cpu().detach().numpy()[0]

    @abstractmethod
    def is_positive(self, target):
        pass

    def roc_eval(self, threshold):
        return self.common_test(threshold)

    def froc_eval(self, threshold, net_number):
        tn, tp, n, p, fp_bboxes = 0, 0, 0, 0, 0
        print('evaluating froc results...')
        for output, target in tqdm(zip(self.outputs[net_number], self.targets)):
            pred = self.gp(output, threshold)
            true = self.gp(target, 0.8)
            # print('pred ', pred)
            # print('true ', true)
            if len(true) == 0:
                continue
            p += 1
            found_true = False
            for pred_bbox in pred:
                for true_bbox in true:
                    if iou(true_bbox[1:], pred_bbox[1:]) > 0.5:
                        found_true = True
                        break
                if not found_true:
                    fp_bboxes += 1
            if found_true:
                tp += 1
            # print('pred: {}'.format(pred))
            # print('true: {}'.format(true))
            # print(tp, tn, p, n)
        return [tp, tn, p, n, fp_bboxes]


    def common_test(self, threshold):
        tn, tp, n, p = 0, 0, 0, 0
        print('evaluating roc results...')
        for output, target in tqdm(zip(self.outputs, self.targets)):
            pred = self.gp(output, threshold)
            if self.is_positive(target):
                p += 1
                if len(pred) > 0:
                    tp += 1
            else:
                n += 1
                if len(pred) == 0:
                    tn += 1
            # print('pred: {}'.format(pred))
            # print('true: {}'.format(true))
            # print(tp, tn, p, n)
        return [tp, tn, p, n]



class SimpleTest(AbstractTest):

    def create_dataset(self):
        luna_test = np.load('./luna_test_{}.npy'.format(self.stage))
        dataset = LungNodule3Ddetector(self.data_path, luna_test, self.config, start=0, end=0, r_rand=self.r_rand)
        return dataset


    def is_positive(self, target):
        return len(self.gp(target, 0.8)) > 0

    def predict_on_data(self, data_loader):
        outputs, targets = [[] for _ in self.nets], []
        for i, (data, target, coord) in enumerate(data_loader):
            data, target, coord = data.cuda(), target.cuda(), coord.cuda()
            data = data.type(torch.cuda.FloatTensor)
            coord = coord.type(torch.cuda.FloatTensor)
            print('data shape: ', data.shape)
            print('coord shape: ', coord.shape)
            # print('coord shape: ', coord.shape)
            for j, net in enumerate(self.nets):
                output = net(data, coord)
                outputs[j].append(output.cpu().detach().numpy()[0])
            targets.append(target)
        return outputs, [self.transform_target(target) for target in targets]


    def test_luna(self, threshold):
        return self.common_test(threshold=threshold)




class PatientTest(AbstractTest):

    def create_dataset(self):
        luna_test = np.load('./luna_test.npy')
        dataset = PatientDataLoader(self.data_path, luna_test, self.config, start=0, end=0)
        return dataset

    def is_positive(self, target):
        return target

    def predict_on_data(self, data_loader):
        outputs, targets = [], []
        print('feeding crops to the net..')
        for i, (data, one_scan_labels, coord) in tqdm(enumerate(data_loader)):
            # print('data shape ', data.shape)
            # print('data shape ', one_scan_labels.shape)
            data = data.transpose(0, 1)
            one_scan_labels = one_scan_labels.transpose(0, 1)
            for crop, label in zip(data, one_scan_labels):
                # print('crop shape ', crop.shape)
                crop, label, coord = crop.cuda(), label.cuda(), coord.cuda()
                crop = crop.type(torch.cuda.FloatTensor)
                coord = coord.type(torch.cuda.FloatTensor)

                output = self.net(crop, coord)
                outputs.append(output.cpu().detach().numpy()[0])
                targets.append(label)
        return outputs, [self.transform_target(target) for target in targets]
