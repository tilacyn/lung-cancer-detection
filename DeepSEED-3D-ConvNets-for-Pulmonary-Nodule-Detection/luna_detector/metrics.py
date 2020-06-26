import pandas as pd
from matplotlib import pyplot as plt
from os.path import join as opjoin
import numpy as np
from path import luna_path

PLOT_SAVE_PATH = opjoin(luna_path, 'plots')
ROC_RESULT_SAVE_PATH = opjoin(luna_path, 'roc_results_npy')


class Metrics:
    def __init__(self,
                 roc_auc,
                 min_dist01,
                 juden_index,
                 sensitivity,
                 specificity,
                 accuracy,
                 precision_positive,
                 precision_negative,
                 positive,
                 negative):
        self.roc_auc = roc_auc
        self.min_dist01 = min_dist01
        self.juden_index = juden_index
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.accuracy = accuracy
        self.precision_positive = precision_positive
        self.precision_negative = precision_negative
        self.positive = positive
        self.negative = negative

    @staticmethod
    def save(metrics, filename):
        df = pd.DataFrame({'roc_auc': [m.roc_auc for m in metrics],
                           'minimal_distance_to_01': [m.min_dist01 for m in metrics],
                           'juden_index': [m.juden_index for m in metrics],
                           'sensitivity': [m.sensitivity for m in metrics],
                           'specificity': [m.specificity for m in metrics],
                           'accuracy': [m.accuracy for m in metrics],
                           'precision_positive': [m.precision_positive for m in metrics],
                           'precision_negative': [m.precision_negative for m in metrics],
                           'positive': [m.positive for m in metrics],
                           'negative': [m.negative for m in metrics],
                           })
        df.to_csv(filename, index=False)


fpr = lambda x: (1 - x[1] / x[3])
tpr = lambda x: x[0] / x[2]
dist01 = lambda x: math.sqrt((fpr(x) ** 2) + ((1 - tpr(x)) ** 2))


def dist_mid(x):
    return (tpr(x) - fpr(x)) / math.sqrt(2)


import math


class MetricsCalculator:
    def __init__(self, roc_result, dice_threshold=0):
        roc_result = list(roc_result.values())
        self.dice_threshold = dice_threshold
        roc_result.reverse()
        self.roc_result = roc_result

    def roc_auc(self):
        roc_auc = 0
        for i in range(len(self.roc_result) - 1):
            roc_auc += (fpr(self.roc_result[i + 1]) - fpr(self.roc_result[i])) * (
                    tpr(self.roc_result[i + 1]) + tpr(self.roc_result[i])) / 2
        return roc_auc

    def draw_roc(self, filename):
        plt.ylim(-0.1, 1.2)
        plt.xlim(-0.1, 1.2)
        plt.ylabel('tpr')
        plt.xlabel('fpr')
        plt.grid()
        mind_idx = np.argmin([dist01(x) for x in self.roc_result])
        mind_x = fpr(self.roc_result[mind_idx])
        mind_y = tpr(self.roc_result[mind_idx])
        print(mind_x, mind_y)
        plt.plot([mind_x, 0], [mind_y, 1], color='green', linestyle='dashed', marker='o')

        juden_idx = np.argmax([dist_mid(x) for x in self.roc_result])
        juden_x = fpr(self.roc_result[juden_idx])
        juden_y = tpr(self.roc_result[juden_idx])
        juden_coord = (juden_x + juden_y) / 2
        print(juden_x, juden_y)
        plt.plot([juden_x, juden_coord], [juden_y, juden_coord], color='purple', linestyle='dashed', marker='o')

        plt.plot([0, 1], [0, 1], color='pink', linestyle='dashed', marker='o')
        plt.plot([0, 1], [1, 1], color='orange', linestyle='dashed')
        plt.plot([1, 1], [0, 1], color='orange', linestyle='dashed')
        print([fpr(x) for x in self.roc_result])
        print([x[0] / x[2] for x in self.roc_result])
        plt.plot([fpr(x) for x in self.roc_result], [x[0] / x[2] for x in self.roc_result], linewidth=3)
        ax = plt.axes()

        ax.arrow(0, 0, 0, 1.1, head_width=0.03, head_length=0.04, fc='k', ec='k', color='blue')
        ax.arrow(0, 0, 1.1, 0, head_width=0.03, head_length=0.04, fc='k', ec='k', color='blue')

        plt.savefig(opjoin(luna_path, filename))
        plt.show()

    def calculate(self, tprw=1):
        res = self.roc_result.copy()
        res.sort(key=lambda x: tprw * tpr(x) - fpr(x))
        optimal = res[-1]
        print(optimal)
        m = Metrics(self.roc_auc(),
                    np.min([dist01(x) for x in self.roc_result]),
                    np.max([dist_mid(x) for x in self.roc_result]),
                    tpr(optimal),
                    1 - fpr(optimal),
                    (optimal[0] + optimal[1]) / (optimal[2] + optimal[3]),
                    optimal[0] / (optimal[0] + optimal[3] - optimal[1]),
                    optimal[1] / (optimal[1] + optimal[2] - optimal[0]),
                    optimal[2],
                    optimal[3])
        return m


class FROCMetricsCalculator:
    def __init__(self, froc_result=None, label=None):
        if froc_result is not None:
            froc_result = list(froc_result.values())
            froc_result.reverse()
            self.roc_result = froc_result
        self.label = label

    def load(self, filename):
        self.label = filename[:-4]
        self.roc_result = np.load(opjoin(ROC_RESULT_SAVE_PATH, filename))

    def draw_roc(self):
        prepare_canvas()
        draw_single_roc(self.roc_result, self.label)

        plt.savefig(opjoin(PLOT_SAVE_PATH, self.label))
        plt.show()

    def save(self, subfolder):
        subfolder = '' if subfolder is None else subfolder
        np.save(opjoin(ROC_RESULT_SAVE_PATH, subfolder, self.label), self.roc_result)


def save_csv(mcs, filename):
    target_xs = [0.25, 0.5, 1, 2, 4, 8]
    results = {}
    for mc in mcs:
        points = get_points(mc.roc_result)
        sensitivities = np.interp(target_xs, points[:, 0], points[:, 1])
        results[mc.label] = sensitivities
        results[mc.label] = [np.round(metric, 3) for metric in results[mc.label]]
    df = pd.DataFrame({'model_name': [label for label in results],
                       '0.25': [results[label][0] for label in results],
                       '0.5': [results[label][1] for label in results],
                       '1': [results[label][2] for label in results],
                       '2': [results[label][3] for label in results],
                       '4': [results[label][4] for label in results],
                       '8': [results[label][5] for label in results],
                       'average': [np.round(np.mean(results[label][:6]), 3) for label in results]
                       })
    df.to_csv(filename, index=False)


def draw_single_roc(roc_result, label):
    points = get_points(roc_result)
    plt.plot(points[:, 0], points[:, 1], linewidth=2, label=label)


def draw_several(mcs, max_x=9.5):
    prepare_canvas(max_x)
    for mc in mcs:
        draw_single_roc(mc.roc_result, label=mc.label)
    plt.legend()
    plt.show()


def get_points(roc_result):
    points = [[res[-1] / res[2], res[0] / res[2]] for res in roc_result]
    points = np.array(points)
    return points


def prepare_canvas(max_x=9.5):
    plt.ylim(-0.1, 1.2)
    plt.xlim(-0.5, max_x)
    plt.ylabel('sensitivity')
    plt.xlabel('average fp / crop')
    plt.grid()
    plt.plot([0, 9], [1, 1], color='pink', linestyle='dashed', marker='o', linewidth=1)
    plt.plot([9, 9], [0, 1], color='pink', linestyle='dashed', marker='o', linewidth=1)
    plt.plot([0.5, 0.5], [0, 1], color='green', linestyle='dashed', linewidth=1)
    plt.plot([1, 1], [0, 1], color='green', linestyle='dashed', linewidth=1)
    plt.plot([2, 2], [0, 1], color='green', linestyle='dashed', linewidth=1)
    plt.plot([4, 4], [0, 1], color='green', linestyle='dashed', linewidth=1)
    plt.plot([8, 8], [0, 1], color='green', linestyle='dashed', linewidth=1)

    ax = plt.axes()

    ax.arrow(0, 0, 0, 1.1, head_width=max_x / 30, head_length=0.04, fc='k', ec='k', color='blue')
    ax.arrow(0, 0, max_x - 0.5, 0, head_width=0.03, head_length=max_x / 30, fc='k', ec='k', color='blue')


def add_averages_to_csv(filename, output):
    df = pd.read_csv(filename)
    metrics = df.to_numpy()

    def create_row(name, type):
        f = None
        if type == 'average':
            f = np.mean
        elif type == 'std':
            f = np.std
        rows = np.array([row for row in metrics if row[0].startswith(name)])
        return ['{}_{}'.format(name, type)] + [np.round(f(values), 3) for values in [rows[:, i + 1] for i in range(7)]]


    average_aug = create_row('augmented', 'average')
    average_baseline = create_row('baseline', 'average')
    std_aug = create_row('augmented', 'std')
    std_baseline = create_row('baseline', 'std')
    df.loc[len(metrics)] = average_aug
    df.loc[len(metrics) + 1] = average_baseline
    df.loc[len(metrics) + 2] = std_aug
    df.loc[len(metrics) + 3] = std_baseline
    df.to_csv(output)
