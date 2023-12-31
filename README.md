# Revisiting-Deformable-Convolution-for-Depth-Completion

This repository is for ReDC method introduced in the following paper accepted by IROS2023:

Revisiting Deformable Convolution for Depth Completion\
Xinglong Sun, Jean Ponce, Yu-Xiong Wang

Arxiv Link: https://arxiv.org/pdf/2308.01905.pdf
## Introduction
Depth completion, which aims to generate highquality
dense depth maps from sparse depth maps, has attracted
increasing attention these years. Previous popular methods
usually employ RGB images as guidance, and introduce iterative
spatial propagation to refine estimated coarse depth
maps. However, most of the propagation refinement methods
require several iterations and suffer from a fixed receptive field,
which may contain irrelevant and useless information with very
sparse input. In this paper, we address these two challenges
simultaneously by revisiting the idea of deformable convolution.
We propose an effective architecture that leverages deformable
kernel convolution as a single-pass refinement module, and
empirically demonstrate its superiority. To better understand
the function of deformable convolution and exploit it for depth
completion, we further systematically investigate a variety of
representative strategies. Our study reveals that, different from
prior work, deformable convolution needs to be applied on an
estimated depth map with a relatively high density for better
performance. We evaluate our model on the large-scale KITTI
dataset and achieve the new state-of-the-art in terms of both
accuracy and inference speed. Our code will be publicly released
upon acceptance.

<div align="center">
  <img src="Figs/flowchart.png" width="100%">
  Overview of our method.
</div>

## Results on KITTI
<div align="center">
  <img src="Figs/result.png" width="100%">
  KITTI Depth Completion Results.
</div>
Link to our public results on KITTI test server:

https://www.cvlibs.net/datasets/kitti/eval_depth_detail.php?benchmark=depth_completion&result=c00c3b4d967f78cb9e1522ebd062f763b7668f7d

## Some Visualizations
<div align="center">
  <img src="Figs/viz1.png" width="100%">
  From left to right: groundtruth dense depth map, guidance RGB image, inference from our model, and inference from PENet.
</div>

<div align="center">
  <img src="Figs/viz2.png" width="100%">
  From left to right: groundtruth dense depth map, guidance RGB image, inference from our model, and inference from PENet.
</div>

## Prerequisites
### Datasets
Please follow the KITTI depth completion dataset downloading instruction here:

https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion

## Train
To train the baseline/unpruned network, run:
```
python3 train.py
```
As mentioned in our paper, we study our deformable refinement module on top of the model backbone based on ENet from the paper PENet (https://arxiv.org/pdf/2103.00783.pdf). For faster convergence of ReDC, you could download the pretrained PENet model from here:
https://drive.google.com/file/d/1RDdKlKJcas-G5OA49x8OoqcUDiYYZgeM/view?usp=sharing

In train.py, we extract the backbone weights from PENet and initialize the backbone used in ReDC.

### Pretrained Models
We also release our pretrained model here: 
https://drive.google.com/file/d/1wE8QLI_fCpGVLKqhqBf5Wg8A08eHJDfv/view?usp=sharing

## Acknowledgement
Some dataloading and evaluation code is from:
https://github.com/JUGGHM/PENet_ICRA2021

## Citations
If you find this repo useful to your project or research, please cite our paper below:

@inproceedings{sun2023revisiting,\
  title={Revisiting deformable convolution for depth completion},\
  author={Sun, X and Ponce, J and Wang, Y-X},\
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems},\
  year={2023}\
}