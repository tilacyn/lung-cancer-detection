import cv2
from models.model_factory import make_model
from os.path import join as opjoin
import numpy as np
from matplotlib import pyplot as plt

def dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return 2 * intersection / (np.sum(y_true) + np.sum(y_pred))


def pad(image):
    return np.pad(image, ((0, 0), (0, 2)), constant_values=1)


class Test:
    def __init__(self, nn_models_dir, predict_threshold):
        self.nn_models_dir = nn_models_dir
        self.predict_threshold = predict_threshold

    def calculate_metrics_for_model(self, model, batches):
        dice_coefficients = []
        for i in range(number_to_show):
            image, image_parts = x[i]
            mask, mask_parts = y[i]
            pred_parts = model.predict(image_parts, batch_size=len(image_parts))
            pred_parts = pred_parts.reshape(16, 16, 256, 256).swapaxes(1, 2).reshape(16 * 256, 16 * 256)
            pred_parts = cv2.resize(pred_parts, (256, 256))
            show(image, mask, pred_parts, self.predict_threshold)

    def calculate_metrics_for_td(self, td, batches):
        if td.metric_eval_timestamp != 'None':
            return td
        else:
            model = make_model(td.model_name, (None, None, 3))
            model.load_weights(opjoin(self.nn_models_dir, td.model_file_name))
            td.add_metrics(self.calculate_metrics_for_model(model, batches))
            return td

    def visualize_for_train_data(self, td, batch, number_to_show=4):
        model = make_model(td.model_name, (None, None, 3))
        model.load_weights(opjoin(self.nn_models_dir, td.model_file_name))
        x, y = batch
        pred = model.predict(x, batch_size=16)
        for i in range(number_to_show):
            image, image_parts = x[i]
            mask, mask_parts = y[i]
            pred_parts = model.predict(image_parts, batch_size=len(image_parts))
            pred_parts = pred_parts.reshape(16, 16, 256, 256).swapaxes(1, 2).reshape(16 * 256, 16 * 256)
            pred_parts = cv2.resize(pred_parts, (256, 256))
            show(image, mask, pred_parts, self.predict_threshold)

def show(x, y, pred, predict_threshold):
    expected = np.reshape(y, (256, 256))
    actual = np.reshape(pred, (256, 256))

    x_to_show = x[:, :, 0]
    x_to_show = x_to_show / x_to_show.max()

    actual = actual > predict_threshold * actual.max()

    to_show = np.array([pad(x_to_show), pad(actual), pad(expected)])
    to_show = np.hstack(to_show.reshape(3, 256, 256))

    plt.imshow(to_show)
    plt.show()

    print('\n===============')
    print('===============\n')
