# MIT License
# 
# Copyright (c) 2019 Yisroel Mirsky
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function, division

import time

from config import *  # user configuration in config.py
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']

from utils.dataloader import DataLoader, SegmentedDataLoader
from tensorflow.keras.layers import Input, Dropout, Concatenate, Cropping3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import UpSampling3D, Conv3D, Dense, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import Constraint
import matplotlib.pyplot as plt
import datetime
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as ktf
import json


# import keras.backend.tensorflow_backend as ktf


def get_session():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


tf.compat.v1.keras.backend.set_session(get_session())


def wasserstein_loss(y_true, y_pred):
    return ktf.mean(y_true * y_pred)


def get_loss(name):
    return wasserstein_loss if name == 'wasserstein' else name

class Trainer:
    def __init__(self, isInjector=True, savepath='default', d_lr=0.0002, combined_lr=0.00001, modelpath=None,
                 generator_weight_updates=1, adain=False, wgan=False, dropout=True, combined_loss=None):
        self.combined_loss = ['wasserstein', 'mse'] if combined_loss is None else combined_loss

        self.generator_weight_updates = generator_weight_updates
        self.isInjector = isInjector
        self.savepath = savepath
        self.adain = adain
        # Input shape
        cube_shape = config['cube_shape']
        self.d_lr = d_lr
        self.combined_lr = combined_lr
        self.img_rows = config['cube_shape'][1]
        self.img_cols = config['cube_shape'][2]
        self.img_depth = config['cube_shape'][0]
        self.channels = 1
        self.num_classes = 5
        self.img_shape = (self.img_rows, self.img_cols, self.img_depth, self.channels)

        # Configure data loader
        if self.isInjector:
            self.dataset_path = config['unhealthy_samples']
            self.modelpath = config['modelpath_inject']
            if modelpath is not None:
                self.modelpath = os.path.join(self.modelpath, modelpath)
            if not os.path.exists(self.modelpath):
                os.makedirs(self.modelpath)
        else:
            self.dataset_path = config['healthy_samples']
            self.modelpath = config['modelpath_remove']

        self.dataloader = SegmentedDataLoader(subset_number=5,
                                              img_res=(self.img_rows, self.img_cols, self.img_depth))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 100
        self.df = 100
        self.wgan = wgan
        self.dropout = dropout

        self.build_combined()
        self.savepath = os.path.join(config['progress'], "injector", self.savepath)
        os.makedirs(self.savepath, exist_ok=True)
        self.save_model_params()

    def save_model_params(self):
        params_path = os.path.join(self.savepath, 'params.json')
        params = {
            'wgan' : self.wgan,
            'dropout' : self.dropout,
            'adain' : self.adain,
            'discriminator lr' : self.d_lr,
            'combined lr' : self.combined_lr,
            'generator weight updates' : self.generator_weight_updates,
            'combined loss' : self.combined_loss
        }
        with open(params_path, 'w') as params_file:
            json.dump(params, params_file, indent=2)


    def build_combined(self):
        optimizer = Adam(self.combined_lr, 0.5)
        optimizer_G = Adam(self.d_lr, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        discriminator_loss = wasserstein_loss if self.wgan else 'mse'
        self.discriminator.compile(
            loss=discriminator_loss,
            optimizer=optimizer_G,
            metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator([img_B])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        combined_loss = [get_loss(name) for name in self.combined_loss]
        self.combined.compile(
            loss=combined_loss,
            loss_weights=[1, 100],
            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def adain(xgb):
            x = xgb[0]
            print('shape before adain: ' + str(x.shape))
            g = xgb[1]
            b = xgb[2]
            mean = ktf.mean(x, axis=[0, 1, 2], keepdims=True)
            std = ktf.std(x, axis=[0, 1, 2], keepdims=True) + 1e-7
            y = (x - mean) / std

            # Reshape gamma and beta
            pool_shape = [-1, 1, 1, 1, y.shape[-1]]
            print(g.shape)
            print(b.shape)
            g = ktf.reshape(g, pool_shape)
            b = ktf.reshape(b, pool_shape)

            print('adain')
            print(y.shape)

            # Multiply by x[1] (GAMMA) and add x[2] (BETA)
            result = y * g + b
            print('shape after adain: ' + str(result.shape))
            return result

        def get_crop_shape(target, refer):
            # depth, the 4rth dimension
            print(target.shape)
            print(refer.shape)
            cd = (target.get_shape()[3] - refer.get_shape()[3])  # .value
            assert (cd >= 0)
            if cd % 2 != 0:
                cd1, cd2 = int(cd / 2), int(cd / 2) + 1
            else:
                cd1, cd2 = int(cd / 2), int(cd / 2)
            # width, the 3rd dimension
            cw = (target.get_shape()[2] - refer.get_shape()[2])  # .value
            assert (cw >= 0)
            if cw % 2 != 0:
                cw1, cw2 = int(cw / 2), int(cw / 2) + 1
            else:
                cw1, cw2 = int(cw / 2), int(cw / 2)
            # height, the 2nd dimension
            ch = (target.get_shape()[1] - refer.get_shape()[1])  # .value
            assert (ch >= 0)
            if ch % 2 != 0:
                ch1, ch2 = int(ch / 2), int(ch / 2) + 1
            else:
                ch1, ch2 = int(ch / 2), int(ch / 2)

            return (ch1, ch2), (cw1, cw2), (cd1, cd2)

        def conv3d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if self.adain and bn:
                g = Dense(filters, bias_initializer='ones')(w)
                b = Dense(filters)(w)
                d = Lambda(adain)([d, g, b])
            elif bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        def deconv3d(layer_input, skip_input, filters, f_size=4, dropout_rate='self'):
            """Layers used during upsampling"""
            u = UpSampling3D(size=2)(layer_input)
            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same')(u)
            if dropout_rate == 'self':
                dropout_rate = 0.5 if self.dropout else False
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            if self.adain:
                g = Dense(filters, bias_initializer='ones')(w)
                b = Dense(filters)(w)
                u = Lambda(adain)([u, g, b])
            else:
                u = BatchNormalization(momentum=0.8)(u)
            u = ReLU()(u)

            # u = Concatenate()([u, skip_input])
            ch, cw, cd = get_crop_shape(u, skip_input)
            crop_conv4 = Cropping3D(cropping=(ch, cw, cd), data_format="channels_last")(u)
            u = Concatenate()([crop_conv4, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape, name="input_image")

        print('input shape')
        print(d0.shape)
        w = Flatten()(d0)
        w = Dense(100, activation='relu')(w)
        w = Dense(100, activation='relu')(w)

        # Downsampling
        d1 = conv3d(d0, self.gf, bn=False)
        d2 = conv3d(d1, self.gf * 2)
        d3 = conv3d(d2, self.gf * 4)
        d4 = conv3d(d3, self.gf * 8)
        d5 = conv3d(d4, self.gf * 8)
        u3 = deconv3d(d5, d4, self.gf * 8)
        u4 = deconv3d(u3, d3, self.gf * 4)
        u5 = deconv3d(u4, d2, self.gf * 2)
        u6 = deconv3d(u5, d1, self.gf)

        u7 = UpSampling3D(size=2)(u6)
        output_img = Conv3D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(inputs=[d0], outputs=[output_img])

    def build_discriminator(self):
        constraint = ClipConstraint(0.01)

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same', kernel_constraint=constraint)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        model_input = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(model_input, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        d5 = Conv3D(1, kernel_size=4, strides=1, padding='same', kernel_constraint=constraint)(d4)

        d6 = Flatten()(d5)
        validity = Dense(1)(d6)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        # valid = np.ones((batch_size,) + self.disc_patch)
        valid = np.ones(batch_size)
        fake = -valid

        g_losses = []
        d_losses_fake = []
        d_losses_original = []
        np.random.seed(int(time.time()))

        for epoch in range(epochs):
            # save model
            if epoch > 0:
                print("Saving Models...")
                self.generator.save(os.path.join(self.modelpath, "G_model.h5"))  # creates a HDF5 file
                self.discriminator.save(
                    os.path.join(self.modelpath, "D_model.h5"))  # creates a HDF5 file 'my_model.h5'

            for batch_i, (imgs_A, imgs_B) in enumerate(self.dataloader.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Condition on B and generate a translated version
                fake_A = self.generator.predict([imgs_B])
                fake_predict = self.discriminator.predict([fake_A, imgs_B])
                original_predict = self.discriminator.predict([imgs_A, imgs_B])

                # Train the discriminators (original images = real / generated = Fake)
                if batch_i % self.generator_weight_updates == np.random.randint(self.generator_weight_updates):
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                    print('d loss fake ', d_loss_fake[0])
                    print('d loss real ', d_loss_real[0])
                    d_losses_fake.append(d_loss_fake[0])
                    d_losses_original.append(d_loss_real[0])
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                else:
                    d_loss = [0, 0]

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                combined1 = self.combined.predict([imgs_A, imgs_B])[0].mean()
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                combined2 = self.combined.predict([imgs_A, imgs_B])[0].mean()
                elapsed_time = datetime.datetime.now() - start_time
                g_loss = np.mean(g_loss)
                g_losses.append(g_loss)
                # Plot the progress
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] [c1: %f, c2: %f, o: %f, f: %f] time: %s" % (
                        epoch, epochs,
                        batch_i,
                        self.dataloader.n_batches,
                        d_loss[0],
                        100 * d_loss[1],
                        g_loss,
                        combined1,
                        combined2,
                        original_predict.mean(),
                        fake_predict.mean(),
                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.show_progress(epoch, batch_i)
                    self.plot_loss(epoch, g_losses, d_losses_fake, d_losses_original)

    def plot_loss(self, epoch, g_losses, d_losses_fake, d_losses_original):
        filename = "loss_%d.png" % (epoch)
        filepath = os.path.join(self.savepath, filename)
        plt.plot(range(len(g_losses)), g_losses, label='gen')
        plt.plot(range(len(d_losses_fake)), d_losses_fake, label='crit_fake')
        plt.plot(range(len(d_losses_original)), d_losses_original, label='crit_orig')
        plt.legend()
        plt.savefig(filepath)
        plt.show()


    def show_progress(self, epoch, batch_i):
        filename = "%d_%d.png" % (epoch, batch_i)
        r, c = 3, 3

        imgs_A, imgs_B = self.dataloader.load_data(batch_size=3)
        fake_A = self.generator.predict([imgs_B])

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(
                    gen_imgs[cnt].reshape((self.img_depth, self.img_rows, self.img_cols))[int(self.img_depth / 2), :,
                    :])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.savepath, filename))
        plt.close()


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return ktf.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}
