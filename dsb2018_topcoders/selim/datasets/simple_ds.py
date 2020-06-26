import numpy as np
from tensorflow.keras.preprocessing.image import Iterator
import time
import os


class SimpleDatasetIterator(Iterator):
    def __init__(self):
        seed = np.uint32(time.time() * 1000)
        super().__init__(100, 5, False, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        return np.random.randn(5, 100, 100, 3), np.random.randn(5, 100, 100, 3)
