import os
from os.path import join as opjoin
import string
import random
import datetime
import json

MODEL_NAME = 'Model Name'
EPOCHS = 'Epochs'
LOSS = 'Loss'
LOSS_VALUE = 'Loss Value'
VAL_DS_LEN = 'Validation Dataset Length'
VAL_STEPS = 'Validation Steps'
GRID_SIZE = 'Grid Size'
CURRENT_TIME = 'Save Time'
RECALL = 'Recall'
SPEC = 'Specificity'
TRAIN_FINISH_TIMESTAMP = 'Training Finished at'
METRIC_EVAL_TIMESTAMP = 'Metrics Calculated at'
MODEL_FILE_NAME = 'Model Filename'
MODELS_JSON = 'models.json'
DICE_COEFFICIENT = 'Dice Coefficient'
TEST_INDEX_LIST_FILE = 'Test index list file'

nn_models_dir = '/content/drive/My Drive/dsb2018_topcoders/selim/nn_models'
models_json_file = opjoin(nn_models_dir, MODELS_JSON)


class TrainData:
    def __init__(self, model_name, epochs, loss, loss_value, val_ds_len, val_steps, grid_size, test_index_list,
                 test_index_list_file=None,
                 train_finish_timestamp=None, metric_eval_timestamp=None, recall=None, spec=None,
                 dice_coefficient=None, model_file_name=None):
        self.model_name = model_name
        self.epochs = epochs
        self.loss = loss
        self.loss_value = loss_value
        self.val_ds_len = val_ds_len
        self.val_steps = val_steps
        self.grid_size = grid_size
        if train_finish_timestamp is None:
            self.train_finish_timestamp = datetime.datetime.now()
        else:
            self.train_finish_timestamp = train_finish_timestamp
        self.test_index_list_file = test_index_list_file
        if test_index_list is None:
            try:
                self.test_index_list = self.load_test_index_list()
            except:
                self.test_index_list = []
        else:
            self.test_index_list = test_index_list
            self.test_index_list_file = self.create_test_index_list_file()
        self.metric_eval_timestamp = metric_eval_timestamp
        self.recall = recall
        self.spec = spec
        self.dice_coefficient = dice_coefficient
        self.models_json = opjoin(nn_models_dir, MODELS_JSON)
        self.model_file_name = model_file_name

    def save(self, model):
        random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        model_file_name = '{}_{}_{}_{}.h5'.format(self.model_name, self.epochs, self.grid_size, random_string)
        self.model_file_name = model_file_name
        model.save(opjoin(nn_models_dir, model_file_name))

        if MODELS_JSON not in os.listdir(nn_models_dir):
            records = []
        else:
            with open(models_json_file, "r") as read_file:
                records = json.load(read_file)
        records.append(self.to_dict())
        with open(models_json_file, "w") as write_file:
            json.dump(records, write_file, indent=1)

    def add_metrics(self, recall=None, spec=None, dice=None):
        self.recall = recall
        self.spec = spec
        self.dice_coefficient = dice
        self.metric_eval_timestamp = datetime.datetime.now()

    def to_dict(self):
        return {
            MODEL_NAME: self.model_name,
            EPOCHS: self.epochs,
            LOSS: self.loss,
            LOSS_VALUE: self.loss_value,
            VAL_DS_LEN: self.val_ds_len,
            VAL_STEPS: self.val_steps,
            GRID_SIZE: self.grid_size,
            RECALL: self.recall,
            SPEC: self.spec,
            DICE_COEFFICIENT: self.dice_coefficient,
            TRAIN_FINISH_TIMESTAMP: str(self.train_finish_timestamp),
            METRIC_EVAL_TIMESTAMP: str(self.metric_eval_timestamp),
            MODEL_FILE_NAME: self.model_file_name,
            TEST_INDEX_LIST_FILE : self.test_index_list_file
        }

    def load_test_index_list(self):
        with open(opjoin(nn_models_dir, self.test_index_list_file), "r") as read_file:
            result = json.load(read_file)
        return result

    def create_test_index_list_file(self):
        random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        file_name = 'test_index_{}.json'.format(random_string)
        with open(opjoin(nn_models_dir, file_name), "w") as write_file:
            self.test_index_list = json.dump(self.test_index_list, write_file)
        return file_name


def from_dict(d):
    return TrainData(d[MODEL_NAME], d[EPOCHS], d[LOSS], d[LOSS_VALUE], d[VAL_DS_LEN], d[VAL_STEPS], d[GRID_SIZE], None,
                     safe_get(d, TEST_INDEX_LIST_FILE), d[TRAIN_FINISH_TIMESTAMP], d[METRIC_EVAL_TIMESTAMP], d[RECALL],
                     d[SPEC], safe_get(d, DICE_COEFFICIENT), d[MODEL_FILE_NAME])


def safe_get(d, key):
    if key in d:
        return d[key]
    else:
        return None
