# ROS-FoundationPose
To run 16GB VRAM GPU, in Pose_Estimation_Model/config/base.yaml;   n_sample_model_point: 256 # Reduced from 1024 was set

## Build Image (tested on RTX 4060TI)
```bash
git clone git@github.com:St333fan/SAM-6D-sasha.git
cd SAM-6D-sasha
docker build --network host -t sam6d_ros .
bash docker/run_container_ros.sh
```
If it's the first time you launch the container, you need to build/install pointnet2. Run this command *inside* the Docker container, every time at startup.
```bash
### Compile pointnet2 every time at docker container start
cd Pose_Estimation_Model/model/pointnet2
python3 -m pip install --upgrade "setuptools>=68" "importlib-metadata>=6"
python3 -m pip install .
```
A running Docker container can be accessed by
```bash
docker exec -it sam6d_ros bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
```

## Setup Data
Download models once in Docker, they will be saved locally
```bash
### Download ISM pretrained model
cd Instance_Segmentation_Model
# python3 download_sam.py
python3 download_fastsam.py
python3 download_dinov2.py
cd ../

### Download PEM pretrained model
cd Pose_Estimation_Model
python3 download_sam6d-pem.py
cd ../
```
## Test Installation
Download test data, without the need for rendering with Blenderporc. Download lmo and put all files from obj_000005 under templates
Downloadable rendered templates [[link](https://drive.google.com/drive/folders/1fXt5Z6YDPZTJICZcywBUhu5rWnPvYAPI?usp=sharing)].
```bash
Data
└── Example
    └── outputs
        ├── sam6d_results
        └── templates
```
```bash
# set the paths for test
export CAD_PATH=/home/stefan/Projects/SAM-6D-sasha/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=/home/stefan/Projects/SAM-6D-sasha/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=/home/stefan/Projects/SAM-6D-sasha/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=/home/stefan/Projects/SAM-6D-sasha/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=/home/stefan/Projects/SAM-6D-sasha/Data/Example/outputs         # path to outputs folder

# run inference
cd SAM-6D
sh demo.sh

# results under
Data/Example/outputs/sam6d_results
```

## ROS start
ROS integration was only tested with Sasha + GraspingPipeline + DOPE setup docker-compose, check this git out:
https://github.com/St333fan/DOPE

# Original README -> <p align="center"> <font color=#008000>SAM-6D</font>: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation </p>

####  <p align="center"> [Jiehong Lin](https://jiehonglin.github.io/), [Lihua Liu](https://github.com/foollh), [Dekun Lu](https://github.com/WuTanKun), [Kui Jia](http://kuijia.site/)</p>
#### <p align="center">CVPR 2024 </p>
#### <p align="center">[[Paper]](https://arxiv.org/abs/2311.15707) </p>

<p align="center">
  <img width="100%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/vis.gif"/>
</p>


## News
- [2024/03/07] We publish an updated version of our paper on [ArXiv](https://arxiv.org/abs/2311.15707).
- [2024/02/29] Our paper is accepted by CVPR2024!


## Update Log
- [2024/03/05] We update the demo to support [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), you can do this by specifying `SEGMENTOR_MODEL=fastsam` in demo.sh.
- [2024/03/03] We upload a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for running custom data.
- [2024/03/01] We update the released [model](https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing) of PEM. For the new model, a larger batchsize of 32 is set, while that of the old is 12. 

## Overview
In this work, we employ Segment Anything Model as an advanced starting point for **zero-shot 6D object pose estimation** from RGB-D images, and propose a novel framework, named **SAM-6D**, which utilizes the following two dedicated sub-networks to realize the focused task:
- [x] [Instance Segmentation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Instance_Segmentation_Model)
- [x] [Pose Estimation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Pose_Estimation_Model)


<p align="center">
  <img width="50%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/overview_sam_6d.png"/>
</p>


## Getting Started

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/JiehongLin/SAM-6D.git
```
Install the environment and download the model checkpoints:
```
cd SAM-6D
sh prepare.sh
```
We also provide a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for convenience.

### 2. Evaluation on the custom data
```
# set the paths
export CAD_PATH=Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=Data/Example/outputs         # path to a pre-defined file for saving results

# run inference
cd SAM-6D
sh demo.sh
```



## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }


## Contact

If you have any questions, please feel free to contact the authors. 

Jiehong Lin: [mortimer.jh.lin@gmail.com](mailto:mortimer.jh.lin@gmail.com)

Lihua Liu: [lihualiu.scut@gmail.com](mailto:lihualiu.scut@gmail.com)

Dekun Lu: [derkunlu@gmail.com](mailto:derkunlu@gmail.com)

Kui Jia:  [kuijia@gmail.com](kuijia@gmail.com)

