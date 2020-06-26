# DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection
DeepSEED: 3D Squeeze-and-Excitation Encoder-Decoder ConvNets for Pulmonary Nodule Detection


Dataset:
LUNA16 can be downloaded from https://luna16.grand-challenge.org/data/

LIDC-IDR can be downloaded from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

-------------------------------------------------------------
### Evaluating the trained model

1. Download model from https://drive.google.com/open?id=10Mm4d_OmBOgJ8eusxC1UeU9ClAndMDGQ

2. Place the trained model file into `luna_detector/test_results` directory

3. Download data for evaluation and training from https://drive.google.com/open?id=13UzuBIXNq_IaEhmE7eSy1LwQ2JXAFukJ

4. Extract the npy files from the `preprocess-result-path.zip` archive downloaded on step 3 and make sure you get the following 
directory `data/preprocess-result-path` with npy files in it. 

5. Go to `luna_detector` folder and run the following code
```python
from test import Test
import numpy as np

result = {}
# PLEASE CHANGE THE THRESHOLD SPACE IF NEEDED
for thr in np.linspace(-1.5, 3, 20):
  # CHANGE path_to_model argument to the relative path of your pretrained model in test_results folder
  ltest = Test(thr=thr, path_to_model='baseline_3/detector_122.ckpt', start=-70, end=0)
  result[thr] = ltest.test_luna()

roc_result = {}
for t in result:
  res = result[t]
  roc_result[t] = res[0]

from matplotlib import pyplot as plt
tnr = []
tpr = []
xs = []
for thr in roc_result:
  r = roc_result[thr]
  xs.append(r[-1] / (r[-2] + r[-3]))
  tpr.append(r[0] / r[2])
  tnr.append(r[1] / r[3])

plt.plot(xs, tpr, label='tpr')
plt.plot(xs, tnr, label='tnr')
plt.ylabel('rate')
plt.xlim(-0.2, 25)
plt.ylim(-0.1, 1.1)
plt.grid()
plt.xlabel('averape fp / scan')
plt.legend()
plt.show()
```

If everything is OK, you will get the ROC curve on a plot.

-------------------------------------------------------------
### Training the model (deprecated)

Preprocessing:
Go to config_training.py, create two directory Luna_data and Preprocess_result_path. Then change directory listed as follows:

Luna_raw: raw data folder downloaded from LUNA16 website

Luna_segment: luna segmentation download from LUNA16 website

Luna_data: temporary folder to store luna data

Preprocess_result_path: final preprocessed data folder

Run prepare.py, output LUNA16 data can be found inside folder Preprocess_result_path, with saved images as _clean.npy, _label.npy for training, and _spacing.npy, _extendbox.npy, _origin.npy are separate information for data testing.



-------------------------------------------------------------
Training:
Go to ./detector directory, the model can be trained by calling the following script:

	CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_detector_se.py -b 16 --save-dir /train_result/ --epochs 150

The output model can be found inside ./train_result/ folder.



-------------------------------------------------------------
Testing:
In order to obtain the predicted label, go to ./detector directory, the model can be tested by calling the following script:

	CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_detector_se.py -b 1 --resume ‘best_model.ckpt’ --test 1 --save-dir /output/

The bbox output can be found inside ./output/bbox/, the predicted bounding boxes and ground truth bounding boxes will be saved in this direction.

Then for FROC metric evaluation, go to FROCeval.py, change path for following directories:

seriesuids_filename: patient ID for testing

detp: threshold for bounding boxes regression

nmsthresh: threshold used bounding boxes non-maximum suppression

bboxpath: directory which stores bounding boxes from testing output

Frocpath: path to store FROC metrics

Outputdir: store the metric evaluation output

The FROC evaluation script is provided from LUNA16 website, you can find the script in noduleCADEvaluationLUNA16.py. 

---------------------------------------------------------------

You could refer to the arxiv paper for more details and performance:

	@misc{li2019deepseed,
	    title={DeepSEED: 3D Squeeze-and-Excitation Encoder-Decoder Convolutional Neural Networks for Pulmonary Nodule Detection},
	    author={Yuemeng Li and Yong Fan},
	    year={2019},
	    eprint={1904.03501},
	    archivePrefix={arXiv},
	    primaryClass={cs.CV}
	}
