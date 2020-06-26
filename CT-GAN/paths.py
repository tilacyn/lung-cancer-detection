from os.path import join as opjoin

my_drive_path = '/content/drive/My Drive'
base_path = opjoin(my_drive_path, 'CT-GAN')
data_base_path = opjoin(my_drive_path, 'DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection')
src_path = opjoin(data_base_path, 'data/luna-data')
annos_path = opjoin(data_base_path, 'luna_detector/labels/annos.csv')