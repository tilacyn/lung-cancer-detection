from os.path import join as opjoin

import numpy as np

from metrics import FROCMetricsCalculator, draw_several, save_csv
from roc_eval import SimpleTest, run_test


def label2model(wa, cv_stage, epoch):
    prefix = 'with_augmented' if wa else 'baseline'
    return {
        'path2model': '{}_128_cv_{}/detector_{}.ckpt'.format(prefix, cv_stage, str(epoch).zfill(3))
    }


def str_label(wa, cv_stage, epoch):
    prefix = 'with_augmented' if wa else 'baseline'
    return '{}_cv_{}_epoch_{}'.format(prefix, cv_stage, str(epoch).zfill(3))


class TestWrapper:
    def __init__(self, wa, cv_stage, epoch):
        self.wa = wa
        self.cv_stage = cv_stage
        self.epoch = epoch
        self.label = str_label(wa, cv_stage, epoch)

    def run(self, r_rand=0.5, stage=0):
        path2model = label2model(self.wa, self.cv_stage, self.epoch)['path2model']
        self.test = SimpleTest(paths2model=[path2model], r_rand=r_rand, stage=stage)

    def eval_metrics(self):
        result = run_test(self.test, mode='froc', left=-1, thr_number=10)
        self.mc = FROCMetricsCalculator(result, label=self.label)

    def load(self, subfolder=None):
        self.mc = FROCMetricsCalculator(label=self.label)
        subfolder = '' if subfolder is None else subfolder
        load_path = opjoin(subfolder, self.label)
        self.mc.load(load_path + '.npy')

    def save(self, subfolder):
        self.mc.save(subfolder)


class DoubleTestWrapper:
    def __init__(self, cv_stage, baseline_epoch, augmented_epoch, all_tta=False):
        self.cv_stage = cv_stage
        self.baseline_epoch = baseline_epoch
        self.augmented_epoch = augmented_epoch
        self.baseline_label = str_label(False, cv_stage, baseline_epoch)
        self.augmented_label = str_label(True, cv_stage, augmented_epoch)
        self.all_tta = all_tta

    def run(self, r_rand=0.5, stage=0):
        path2baseline = label2model(False, self.cv_stage, self.baseline_epoch)['path2model']
        path2augmented = label2model(True, self.cv_stage, self.augmented_epoch)['path2model']
        self.test = SimpleTest(paths2model=[path2baseline, path2augmented], r_rand=r_rand, stage=stage, all_tta=self.all_tta)

    def eval_metrics(self):
        baseline_result = run_test(self.test, mode='froc', left=-1, thr_number=10, net_number=0)
        augmented_result = run_test(self.test, mode='froc', left=-1, thr_number=10, net_number=1)
        self.baseline_mc = FROCMetricsCalculator(baseline_result, label=self.baseline_label)
        self.augmented_mc = FROCMetricsCalculator(augmented_result, label=self.augmented_label)

    def load(self, subfolder=None):
        self.baseline_mc = FROCMetricsCalculator(label=self.baseline_label)
        subfolder = '' if subfolder is None else subfolder
        baseline_load_path = opjoin(subfolder, self.baseline_label)
        self.baseline_mc.load(baseline_load_path + '.npy')

        self.augmented_mc = FROCMetricsCalculator(label=self.augmented_label)
        aug_load_path = opjoin(subfolder, self.augmented_label)
        self.augmented_mc.load(aug_load_path + '.npy')

    def save(self, subfolder):
        self.baseline_mc.save(subfolder)
        self.augmented_mc.save(subfolder)



class TripleTestWrapper(DoubleTestWrapper):
    def add_baseline(self, path2model):
        self.path2second_baseline = path2model

    def run(self, r_rand=0.5, stage=0):
        path2baseline = label2model(False, self.cv_stage, self.baseline_epoch)['path2model']
        path2augmented = label2model(True, self.cv_stage, self.augmented_epoch)['path2model']
        self.test = SimpleTest(paths2model=[path2baseline, path2augmented, self.path2second_baseline], r_rand=r_rand, stage=stage)

    def eval_metrics(self):
        baseline_result = run_test(self.test, mode='froc', left=-1, thr_number=10, net_number=0)
        augmented_result = run_test(self.test, mode='froc', left=-1, thr_number=10, net_number=1)
        self.baseline_mc = FROCMetricsCalculator(baseline_result, label=self.baseline_label)
        self.augmented_mc = FROCMetricsCalculator(augmented_result, label=self.augmented_label)
        second_baseline_result = run_test(self.test, mode='froc', left=-1, thr_number=10, net_number=2)
        second_baseline_mc = FROCMetricsCalculator(second_baseline_result, label='label')
        self.baseline_mc.roc_result = np.add(self.baseline_mc.roc_result, second_baseline_mc.roc_result)


class CrossValidation:
    def __init__(self, subfolder, all_tta=False):
        self.stage2tw = {}
        self.subfolder = subfolder
        self.all_tta = all_tta

    def init_stage(self, number, baseline_epoch, wa_epoch):
        tw = DoubleTestWrapper(number, baseline_epoch, wa_epoch, self.all_tta)
        self.stage2tw[number] = tw

    def init_triple_stage(self, number, baseline_epoch, wa_epoch,  path2model):
        tw = TripleTestWrapper(number, baseline_epoch, wa_epoch, self.all_tta)
        tw.add_baseline(path2model)
        self.stage2tw[number] = tw

    def run_stage(self, number, r_rand=0.5):
        self.stage2tw[number].run(r_rand, number)

    def eval_stage_metrics(self, number):
        self.stage2tw[number].eval_metrics()


    def save_stage(self, number):
        self.stage2tw[number].save(self.subfolder)

    def load_stage(self, number):
        self.stage2tw[number].load(self.subfolder)

    def run_stages(self):
        for i in range(5):
            self.run_stage(i)

    def save_stages(self):
        for i in range(5):
            self.save_stage(i)

    def load_stages(self):
        for i in range(5):
            self.load_stage(i)

    def draw(self, stage_n):
        tw = self.stage2tw[stage_n]
        draw_several([tw.baseline_mc, tw.augmented_mc])

    def save_csv(self, filename):
        mcs = []
        for tw in self.stage2tw.values():
            mcs.append(tw.baseline_mc)
            mcs.append(tw.augmented_mc)
        save_csv(mcs, filename)

    def rename_labels(self):
        for i in range(5):
            self.stage2tw[i].baseline_mc.label = 'baseline_{}'.format(i + 1)
            self.stage2tw[i].augmented_mc.label = 'augmented_{}'.format(i + 1)