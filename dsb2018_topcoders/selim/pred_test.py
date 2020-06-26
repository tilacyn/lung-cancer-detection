import os

from params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from models.model_factory import make_model

from datasets.lidc import imread
from os import path, mkdir, listdir

import timeit
import cv2
from tqdm import tqdm
import numpy as np

test_folder = args.test_folder

all_ids = []
all_images = []
all_masks = []

OUT_CHANNELS = args.out_channels

def preprocess_inputs(x):
    return preprocess_input(x, mode=args.preprocessing_function)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    weights = [os.path.join(args.models_dir, m) for m in args.models]
    models = []
    for w in weights:
        model = make_model(args.network, (None, None, 3))
        print("Building model {} from weights {} ".format(args.network, w))
        model.load_weights(w)
        models.append(model)
    print('Predicting test')
    for d in tqdm(listdir(test_folder)):
        print('processing {}'.format(d))
        img = imread(test_folder + '/' + d)[0]

        pred = models[0].predict(np.array([img], 'float32'), batch_size=1)
        print(pred.shape)
        cv2.imwrite(os.path.join(args.out_root_dir, d + '.jpg'), pred[0][:,:,1])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))