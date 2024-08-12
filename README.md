# Guard-Net

## Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Environment](#environment)
  - [Dependencies](#dependencies)
- [Demo](#demo)
- [Training](#training)
  - [Data Preparation](#data-preparation)
  - [SceneFlow](#sceneflow)
  - [KITTI](#kitti)
- [Testing](#testing)
- [Acknowledgments](#acknowledgments)

## Introduction

------

The code of the paper [Guard-Net](https://ieeexplore.ieee.org/abstract/document/10433888).

## Installation
------
### Environment

Create a virtual environment and activate it

```bash
conda create -n guard python=3.8
source activate guard
```
### Dependencies

install PyTorch (An example works for me)

 `Link`：https://pan.baidu.com/s/1aUj3kB3wMkhFigTYdyHYGg?pwd=6acz 

`Code`：6acz

```bash
pip install torchvision-0.10.1+cu111-cp38-cp38-linux_x86_64.whl
pip install torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl
```
install extra dependencies

```bash
pip install -r requirements.txt
```

## Demo

------

Generate disparity images of SceneFlow test set: 

`Download SceneFlow Pre-trained Weights`

 `Link`：https://pan.baidu.com/s/1gbIT8CMvHTJiIMyJLFh67w?pwd=9d65 

`Code`：9d65 

```bash
python ./save_disp_sceneflow.py \
    --datapath SceneFlow Path \
    --loadckpt SceneFlow Pre-trained Weights Path
```

## Training

------
### Data Preparation
* [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

* [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)

* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

In our setup, the dataset is organized as follows
```
└── datasets
    ├── KITTI
    |   ├── 2012
    |   │   ├── training
    |   │   ├── testing
    |   |
    |   ├── 2015
    |   │   ├── training
    |   │   └── testing
    |
    └── SceneFlow
    	├── flyingthings3d__frames_finalpass
        ├── flyingthings3d__disparity
	├── driving__frames_finalpass
        ├── driving__disparity
        ├── flyingthings3d_final
        ├── monkaa__frames_finalpass
        ├── monkaa__disparity
```


### SceneFlow
Use the following command to train Guard-Net on SceneFlow

```bash
python ./main_coex.py \
    --datapath SceneFlow Path \
    --logdir logdir Path
```
### KITTI
Use the following command to train Guard-Net on KITTI (using pre-trained model on Scene Flow)

```bash
python ./coex_kitti.py \
    --kitti15_datapath kitti15 Path \
    --kitti12_datapath kitti12 Path \
    --logdir logdir Path \
    --loadckpt SceneFlow Pre-trained Weights Path
```

## Testing

------

Use the following command to test Guard-Net on SceneFlow

```bash
python ./test_sceneflow.py \
    --datapath SceneFlow Path \
    --loadckpt SceneFlow Pre-trained Weights Path
```

## Acknowledgments

Thanks to Antyanta Bangunharcana for open-sourcing his excellent work [CoEx](https://github.com/antabangun/coex). Thanks to Gangwei Xu for open-sourcing his PyTorch implementation. 
