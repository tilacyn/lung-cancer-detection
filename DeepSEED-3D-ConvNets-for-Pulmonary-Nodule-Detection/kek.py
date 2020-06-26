
import numpy as np
import glob
import pandas

npy_files = glob.glob('data/preprocess-result-path/*label.npy')
a = np.load(npy_files[0])

luna_label = 'data/annotations.csv'

# for f in npy_files:
#     print(np.load(f))

annos = np.array(pandas.read_csv(luna_label))
