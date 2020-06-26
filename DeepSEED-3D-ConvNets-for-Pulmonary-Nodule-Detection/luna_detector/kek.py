import glob
from importlib import import_module

import numpy as np
from data_loader import LungNodule3Ddetector
from lidc_dataset import LIDCDataset

from config_training import config as config_training

data_dir = 'data/preprocess-result-path'
luna_train = np.load('./luna_detector/luna_train.npy')

idcs = glob.glob('./data/preprocess-result-path/*label.npy')

idcs = [int(idc[-13:-10]) for idc in idcs]

idcs = [idc for idc in idcs if idc > 100]

model = import_module('res18_se')
print('creating model')
config, net, loss, get_pbb = model.get_model()
print('created model')

# ds = LungNodule3Ddetector(data_dir, idcs, config, phase='train')

ds = LIDCDataset(data_dir, config)

a = ds[0]