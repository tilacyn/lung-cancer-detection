import gc
import cv2
import tensorflow

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os
from params import args as args_set

os.environ['CUDA_VISIBLE_DEVICES'] = args_set.gpu

from aug.transforms import aug_mega_hardcore

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import S
from tensorflow.keras.utils import multi_gpu_model

from tensorflow.losses import

from datasets.lidc import LIDCDatasetIterator
from datasets.lidc import LIDCValidationDatasetIterator
from datasets.simple_ds import SimpleDatasetIterator
from models.model_factory import make_model
from models.unets import custom

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from losses import make_loss, hard_dice_coef, hard_dice_coef_ch1

from tensorflow.python.client import device_lib

import tensorflow.keras.backend as K
import numpy as np

class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


gpus = [x.name for x in device_lib.list_local_devices() if x.name[:4] == '/gpu']


def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index + 1]:
            l.trainable = False


def run(args):
    if args.crop_size:
        print('Using crops of shape ({}, {})'.format(args.crop_size, args.crop_size))
    else:
        print('Using full size images')
    folds = [int(f) for f in args.fold.split(",")]
    for fold in folds:
        # model = make_model(args.network, (None, None, channels))
        model = custom()
        if args.weights is None:
            print('No weights passed, training from scratch')
        else:
            weights_path = args.weights.format(fold)
            print('Loading weights from {}'.format(weights_path))
            model.load_weights(weights_path, by_name=True)
        # freeze_model(model, args.freeze_till_layer)
        optimizer = RMSprop(lr=args.learning_rate)
        print('learning rate: {}'.format(args.learning_rate))
        if args.optimizer:
            if args.optimizer == 'rmsprop':
                optimizer = RMSprop(lr=args.learning_rate, decay=float(args.decay))
            elif args.optimizer == 'adam':
                optimizer = Adam(lr=args.learning_rate, decay=float(args.decay))
            elif args.optimizer == 'amsgrad':
                optimizer = Adam(lr=args.learning_rate, decay=float(args.decay), amsgrad=True)
            elif args.optimizer == 'sgd':
                optimizer = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True, decay=float(args.decay))
        train_generator = LIDCDatasetIterator(args.images_dir, args.batch_size)
        best_model_file = '{}/best_{}{}_fold{}.h5'.format(args.models_dir, args.alias, args.network, fold)

        best_model = ModelCheckpointMGPU(model, filepath=best_model_file, monitor='val_loss',
                                         verbose=1,
                                         mode='min',
                                         period=args.save_period,
                                         save_best_only=True,
                                         save_weights_only=True)
        last_model_file = '{}/last_{}{}_fold{}.h5'.format(args.models_dir, args.alias, args.network, fold)

        last_model = ModelCheckpointMGPU(model, filepath=last_model_file, monitor='val_loss',
                                         verbose=1,
                                         mode='min',
                                         period=args.save_period,
                                         save_best_only=False,
                                         save_weights_only=True)
        if args.multi_gpu:
            model = multi_gpu_model(model, len(gpus))
        model.compile(
            # loss=make_loss('double_head_loss'),
            loss='mse',
            # loss=make_loss(args.loss_function),
            optimizer=optimizer,
            metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef]
        )

        def schedule_steps(epoch, steps):
            for step in steps:
                if step[1] > epoch:
                    print("Setting learning rate to {}".format(step[0]))
                    return step[0]
            print("Setting learning rate to {}".format(steps[-1][0]))
            return steps[-1][0]

        callbacks = [best_model, last_model]

        if args.schedule is not None:
            steps = [(float(step.split(":")[0]), int(step.split(":")[1])) for step in args.schedule.split(",")]
            lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, steps))
            callbacks.insert(0, lrSchedule)
        tb = TensorBoard("logs/{}_{}".format(args.network, fold))
        callbacks.append(tb)
        # validation_data = LIDCValidationDatasetIterator(args.images_dir, args.batch_size)
        # validation_steps = validation_data.n

        print(model.summary())

        def gen():
            while 1:
                yield np.full((1, 256, 256, 3), 100), np.full((1, 10, 10, 2), 240)

        ds = tensorflow.data.Dataset.from_generator(gen,
                                                    (tensorflow.int8,  tensorflow.int8),
                                                    (tensorflow.TensorShape([1, 256, 256, 3]), tensorflow.TensorShape([1, 10, 10, 2])))

        model.fit(
            ds,
            # train_generator,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            # validation_data=validation_data,
            # validation_steps=validation_steps,
            # callbacks=callbacks,
            max_queue_size=5,
            verbose=1,
            workers=args.num_workers)

        print(model.predict(np.full((1, 256, 256, 3), 100), 1))

        model.save

        del model
        K.clear_session()
        gc.collect()

        model


if __name__ == '__main__':
    run(args_set)
