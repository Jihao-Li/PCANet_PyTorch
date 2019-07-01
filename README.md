# PCANet_PyTorch
This is a PyTorch implementation of PCANet. Details are described in [the original paper](https://arxiv.org/abs/1404.3606).  
Unlike other implementations, the number of stages in PCANet can be set arbitrarily, rather than two. So the structure is more flexible. 

# Requirements
* Python 3.5
* PyTorch==1.0.0
* sklearn, tensorboardX, numpy

# Usage
train
----------
`python train.py`

eval
----------
`python eval.py --pretrained_path <path to trained PCANet model and SVM>`
  
# Results on MNIST
use 70% of training data
----------
convolution kernel in stage 0  
![](https://github.com/JihaoLee/PCANet_PyTorch/blob/master/results/kernel0_trainset0.7.png)  
convolution kernel in stage 1  
![](https://github.com/JihaoLee/PCANet_PyTorch/blob/master/results/kernel1_trainset0.7.png)  
feature maps of an image in stage 0  
![](https://github.com/JihaoLee/PCANet_PyTorch/blob/master/results/feature0_trainset0.7.png)  
feature maps of an image in stage 1  
![](https://github.com/JihaoLee/PCANet_PyTorch/blob/master/results/feature1_trainset0.7.png)  
**the accuracy rate in total testing data is 93.42%**
  
# Results on cifar10
in progress

# Citation
Chan T H , Jia K , Gao S , et al. PCANet: A Simple Deep Learning Baseline for Image Classification?[J]. IEEE Transactions on Image Processing, 2015, 24(12):5017-5032.
