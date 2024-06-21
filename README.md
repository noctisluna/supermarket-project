# Yolov5_ros

This package provides a ROS wrapper for [PyTorch-YOLOv5](https://github.com/ultralytics/yolov5) based on PyTorch-YOLOv5. The package has been tested with Ubuntu 16.04 and Ubuntu 18.04.

# develop environment：
- Ubuntu 16.04 / 18.04
- ROS Kinetic / Melodic
- Python>=3.6.0 environment, including PyTorch>=1.7

# Prerequisites:

## Install Anaconda:

### 1. First download the corresponding installation package [Anaconda](https://www.anaconda.com/products/individual#linux)
### 2. Then install anaconda （for example）

```
bash ~/Downloads/Anaconda3-2024.02-1-Linux-x86_64.sh
```

## Install Pytorch:

### 1. First create an anaconda virtual environment for pytorch

```
conda create -n mypytorch python=3.8
```
### 2. activate the mypytorch environment

```
conda activate mypytorch
```
### 3. Select the specified version to install Pytorch on the PyTorch official website

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Installation yolov5_ros

```
cd /your/catkin_ws/src

git clone https://github.com/noctisluna/supermarket-project.git

cd supermarket-project/yolov5_ros/yolov5

pip install -r requirements.txt
```

## Basic Usage

1. First, if you train your own model, make sure to put your weights in the [weights](https://github.com/noctisluna/supermarket-project/tree/main/yolov5_ros/weights) folder. The default weights loaded here are the optimal weights “best.pt” we trained.
2. The default settings (using `best.pt`) in the `launch/yolo_v5.launch` file should work, all you should have to do is change the image topic you would like to subscribe to.
3. You may need to install (if not already installed) additional libraries such as OpenCV, playsound, Speech_Recognition and gTTS (Google Text-to-Speech) to meet the dependencies of this project.
4. After installing the required environment for this project, you only need to run the following command in the terminal to run this project in an integrated manner. You need to voice input "start" to start the robot, and correspondingly, use "stop" to shut down the robot.

```
roslaunch yolov5_ros yolo_v5.launch
```


### Statements

Please make sure your camera and microphone are properly connected to the robot. Due to hardware or environmental factors, you may need to input multiple voice commands until the robot correctly recognizes your "start" and "stop" commands.


    


