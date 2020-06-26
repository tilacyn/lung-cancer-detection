### Augmenting LIDC Dataset Using 3D Generative Adversarial Networks to Improve Lung Nodule Detection

* 2019

* code on github

* link https://arxiv.org/pdf/1904.05956.pdf

* adopted Wasserstein GAN

* adopted CNN model from this paper https://www.researchgate.net/publication/323440759_Lung_nodule_detection_from_CT_scans_using_3D_convolutional_neural_networks_without_candidate_selection
(Access restricted unfortunately)

* results: (The number of FP seems pretty weird which is explained in paper as that real FP rate has been multiplied with (total FP number / total TP number), and that also seems strange because in RPN for example this asymptotically equal to FP^2 (TP is always O(1) per image))

![Tux, the Linux mascot](/Users/mkryuchkov/CT-GAN/3.png)

### CT-GAN: Malicious Tampering of 3D Medical Imagery using Deep Learning

* 2019

* code on github

* provides nodule injection / removal framework

* no training or evaluation of lung nodule detection have been performed 

* used conditional GANs (No more specific information about model did I manage to get from the paper)

* link: https://www.usenix.org/system/files/sec19-mirsky_0.pdf


### Synthesizing Diverse Lung Nodules Wherever Massively: 3D Multi-Conditional GAN-based CT Image Augmentation for Object Detection

* 2019

* did not find code on github

* link: https://arxiv.org/pdf/1906.04962.pdf 

* adopted Multi-Conditional GAN

* adopted 3d Faster RCNN

* results (look less weird than in the fisrt work):

![Tux, the Linux mascot](/Users/mkryuchkov/CT-GAN/1.png)

![Tux, the Linux mascot](/Users/mkryuchkov/CT-GAN/2.png)
