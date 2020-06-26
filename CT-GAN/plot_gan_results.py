import os

import numpy as np
import tensorflow.keras.backend as ktf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from utils.dataloader import SegmentedDataLoader


class NodulePlot:
    def __init__(self, model_path, save_path):
        self.dataloader = SegmentedDataLoader(subset_number=5, img_res=(32, 32, 32))
        self.generator = load_model(os.path.join(model_path, "G_model.h5"),
                                    custom_objects={'ktf': ktf})
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def plot(self, filename):
        r, c = 3, 3

        imgs_A, imgs_B = self.dataloader.load_data(batch_size=3)
        fake_A = self.generator.predict([imgs_B])

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c + 1)
        cnt = 0
        for i, title in zip(range(r), titles):
            axs[i, 0].text(0.35, 0.5, title)
            axs[i, 0].axis('off')
            for j in range(c):
                axs[i, j + 1].imshow(
                    gen_imgs[cnt].reshape((32, 32, 32))[16, :, :])
                axs[i, j + 1].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.save_path, filename))
        plt.close()
