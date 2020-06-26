import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy
import cv2
from tensorflow import convert_to_tensor
import tensorflow as tf
import math
import numpy as np


def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hard_dice_coef_ch1(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice


def binary_crossentropy(y, p):
    # print(y.eval())
    # print(p.eval())
    result = K.mean(K.binary_crossentropy(y, p))
    # print(result)
    # raise NotImplementedError
    return result


def double_head_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    # contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    return mask_loss


def mask_contour_mask_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    full_mask = dice_coef_loss_bce(y_true[..., 2], y_pred[..., 2])
    return mask_loss + 2 * contour_loss + full_mask


def softmax_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) * 0.6 + dice_coef_loss(y_true[..., 0],
                                                                           y_pred[..., 0]) * 0.2 + dice_coef_loss(
        y_true[..., 1], y_pred[..., 1]) * 0.2


def make_loss(loss_name, c=None):
    if loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.5, bce=0.5)

        return loss
    elif loss_name == 'bce':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0, bce=1)

        return loss
    elif loss_name == 'categorical_dice':
        return softmax_dice_loss
    elif loss_name == 'double_head_loss':
        return double_head_loss
    elif loss_name == 'mask_contour_mask_loss':
        return mask_contour_mask_loss
    elif loss_name == 'custom_mse':
        return lambda x, y: custom_mse(c, x, y)
    else:
        ValueError("Unknown loss.")


def tsr(arr):
    return convert_to_tensor(arr, dtype=tf.float32)


def custom_bce(y_true, y_pred):
    result = []
    for i in range(len(y_pred)):
        y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_pred[i]]
        result.append(-np.mean(
            [y_true[i][j] * math.log(y_pred[i][j]) + (1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in
             range(len(y_pred[i]))]))
    return np.mean(result)


def test():
    img1 = cv2.imread('/Users/mkryuchkov/lung-ds/000001.jpg')
    img2 = cv2.imread('/Users/mkryuchkov/lung-ds/000002.jpg')
    return make_loss('bce_dice')(convert_to_tensor(img1, dtype=tf.float32), convert_to_tensor(img1, dtype=tf.float32))


def custom_mse(c, y_true, y_pred):
    return c * K.mean(
        K.square(tf.where(tf.greater(y_true, y_pred), y_pred, tf.zeros([256, 256, 2])) - y_true))
        #    + K.mean(
        # K.square(y_true - y_pred))
