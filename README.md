# Open-Source Vision-Based Driver Assist for Dashcams
<p align="center"> <img src='readme/obsandlane.gif' align="center" height="480px"> 
  
## Features

- Object Detection and Tracking
- Lane Detection and Centering Error Calculation
- Speed Estimation (needs integration)
- Top-down birds-eye view

FreeDAS is an open-source project developed for our Computer Vision & Computational Photography course as a final project. The goal of this project is to enable old vehicles with no access to driver assist functionalities with a vision-based driver assist system using webcam footage. This is now an initial proof of concept and the above features have been implemented, with special techniques to enable faster computation.

Our inference code is based on [the official CenterNet repo](https://github.com/xingyizhou/CenterNet), albeit heavily modified to enable our own custom choice of the [COCO dataset](https://cocodataset.org/#home) classes as well as integration with other functionalities in our code. 

## Installation
*Our system was tested on a GigaByte Aorus 15P-YD laptop with an NVIDIA RTX 3080 mobile and an Intel i7-11800H running Windows. We do not guarantee operation on non-windows environments or AMD GPUs.*
#### Cloning the Code
The first step of installation is to clone this repository locally. When that is done, make sure you have DCNv2 cloned as well. This can be easily done by running the command below in the main directory:
```
git submodule --init --update
```
This should clone the DCNv2 repository into FreeDAS/src/lib/models/networks/. The reason we are fetching DCNv2 externally instead of utilizing DCNv2 that is packaged with CenterNet is due to incompatibility with windows. For more information, please refer to [this github issue](https://github.com/xingyizhou/CenterNet/issues/7).

#### Installing Prerequisites
To be able to use our code, please ensure that you first have all prerequisites installed. To do this, simply run the following command:
```
pip install -r requirements.txt
```
"requirements.txt" refers to the text file in the main directory.
You will also require the following tools to be able to build DCNv2:

- [CUDA](https://developer.nvidia.com/cuda-downloads) (11.5 tested) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
- [CMake](https://cmake.org/download/) (3.22.0 tested)
- [Visual Studio 2019 Build Tools](https://visualstudio.microsoft.com/vs/older-downloads/) 2019 is crucial for correct building of this code. If you have the 2022 version, building DCNv2 WILL FAIL.
- [Python](https://pytorch.org/) (3.9.9 tested)
- [tensorflow 1.7.1](https://www.tensorflow.org/install/pip#3.-install-the-tensorflow-pip-package) This version of tensorflow is crucial as newer versions also cause DCNv2 to fail the build.

#### Building DCNv2
Now that we have DCNv2 cloned in our repository, we need to build DCNv2 to be able to import it into our code. This can be done by running the following:
```
cd FreeDAS\src\lib\models\networks\DCNv2

python setup.py build develop
```
## Accessing CenterNet Models
To access models used in this code, please use [this drive link](https://drive.google.com/drive/folders/1l3nlYQu2W2VkrBAMXB0dXvi6bFceBnZf?usp=sharing). These include both CenterNet models referenced in our run scripts as well as the speed estimation model. Again, we take no credit for CenterNet models as they were neither developed nor trained by us. You can get the latest versions of these models from [CenterNet's official model zoo](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md)
## Running the Code
#### Object Detection
To run the main code, simply run the following command in the main folder:
```
python src/demo.py ctdet --demo Data/train.mp4 --load_model models/ctdet_coco_hg.pth --arch hourglass
```
This will run our code utilizing CenterNet's HourGlass architecture for object detection. This will provide the most stable and most accurate detections. Note that this model takes 600MB of VRAM and can be quite computationally intensive to run. As such, we have provided different examples for running the code in FreeDAS\run. These four other models are smaller and less expensive to run, so you might need to use those models for better performance. Note that this might result in poorer object detection when compared to *"ctdet_coco_hg.pth"*

The four alternative run commands are provided below:

**DLA 1X**
```
python src/demo.py ctdet --demo Data/train.mp4 --load_model models/ctdet_coco_dla_1x.pth
```
**DLA 2X**
```
python src/demo.py ctdet --demo Data/train.mp4 --load_model models/ctdet_coco_dla_2x.pth
```
**ResNet18 DCN**
```
python src/demo.py ctdet --demo Data/train.mp4 --load_model models/ctdet_coco_resdcn18.pth --arch resdcn_18
```
**ResNet101 DCN**
```
python src/demo.py ctdet --demo Data/train.mp4 --load_model models/ctdet_coco_resdcn101.pth --arch resdcn_101
```
For more information on why these commands are used, please refer to CenterNet's officail documentation. It is crucial that you download the models referenced in the previous section and saving them in FreeDAS\models prior to runninng this code, else it will fail.
#### Speed Estimation
Speed Estimation is currently as stand-alone feature as we were unable to load both CenterNet and our trained model together without running out of memory. This is clear indication that the model needs further optimization before integration. However, you can still run the speed estimation model by simply running the following command:
```
python src\lib\SpeedEstimator.py
```
You should get something similar to this:

<p align="center"> <img src='readme/Speedestimatorvideo.gif' align="center" height="480px"> 
