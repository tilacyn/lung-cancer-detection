import numpy as np
from os.path import join as opjoin
import os
import pandas as pd
from prepare import savenpy_luna, load_itk_image
from path import augmented_prp, annos_path, segment_path

class AugmentationService:
    def __init__(self, augmented_data_path):
        self.augmented_data_path = augmented_data_path
        self.annos = pd.read_csv(annos_path).to_numpy()


    def load_np(self, scan_id):
        path2scan = opjoin(self.augmented_data_path, 'generated_{}.mhd.npy'.format(scan_id))
        scan = np.load(path2scan)
        mask, origin, spacing, is_flip  = load_itk_image(opjoin(segment_path, '{}.mhd'.format(scan_id)))
        return scan, origin, spacing, is_flip, mask, scan_id


    # preprocess augmented
    def preprocess_augmented_data(self):
        augmented_scan_paths = [path for path in os.listdir(self.augmented_data_path) if path.endswith('mhd.npy')]
        scan_ids = [path[-11:-8] for path in augmented_scan_paths]
        for scan_id in scan_ids:
            if os.path.exists(os.path.join(augmented_prp, '{}_clean.npy'.format(scan_id))):
                print('skipping {}'.format(scan_id))
                continue
            try:
                savenpy_luna(scan_id, self.annos, None, None, None, augmented_prp, self.load_np)
            except:
                pass
