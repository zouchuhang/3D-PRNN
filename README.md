# 3D-PRNN
Torch implementation of ICCV 17 [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zou_3D-PRNN_Generating_Shape_ICCV_2017_paper.pdf): "3D-PRNN, Generating Shape Primitives with Recurrent Neural Networks"

For academic use only.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN
- Torch
  
  matio: https://github.com/tbeu/matio
  
  distributions: https://github.com/deepmind/torch-distributions

## Data
- download primitive data to current folder
```
wget http://czou4.web.engr.illinois.edu/data/data_3dp.zip
```
  
This includes 

## Train

## Generation

## Visualization

## Note


## Citation
```
@InProceedings{Zou_2017_ICCV,
author = {Zou, Chuhang and Yumer, Ersin and Yang, Jimei and Ceylan, Duygu and Hoiem, Derek},
title = {3D-PRNN: Generating Shape Primitives With Recurrent Neural Networks},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```

## Acknowledgement
- We express gratitudes to the torch implementation of [hand writting digits generation](https://github.com/jarmstrong2/handwritingnet) as we benefit a lot from the code.
