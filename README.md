# Graph-Based-Object-Detection-on-Pointclouds

## Summary of our work

In this work we attempt to develop a model for the task of 3D object detection on lidar pointclouds collected for the purpose of development and improvement functionality of self-driving cars such as the task of navigation. 

Due to the sparse nature of pointclouds, we, in this work attempt to leverage the ablity of graph convolutional networks to connect related data points, to obtain spatial encodings of the pointcloud.

The main intention behind this idea was to develop a network with an architecture similar to the convolutional neural network architecture which is popularly used for the various image processing tasks. To this end we progressively sample the pointclouds at each layer using farthest point sampling to reduce the size while also encoding the spatial features of the pointclouds at every stage to make sure that information regarding the structure of the pointcloud is not lost during any sort of sampling. The features of the remaining points after the end of sampling are then passed through separate convolution blocks to predict the class of the point and the 3D bounding box parameters of the object of which the point is a part of.

## Requirements 

To run this code base, the following python packages would be required:
- torch
- torchvision
- setuptools
- numpy
- tqdm

**CUDA IS NEEDED TO RUN THIS CODE.** This is because the certain functionalities have such as fathest point sampling and iou calculation which are used in the code base are written in cuda so as to reduce processing time.

## Codebase and Folder structure

The following folder structure is assumed by the code base during training.
- **root**
  - **dataset** : This folder should contain the dataset on which the model needs to be trained
    - **calib** : This folder should contain the calibration values to convert the coordinate system of the pointcloud from camera sensor to lidar sensor and vise versa
    - **label_2** : This folder should contain the annotations for the pointclouds in the format specified in the kitti dataset
    - **velodyne** : This folder should contain the pointcloud files as .bin files.
  - **iou3d** : This folder contains the code for calculating the 3D IoU of the bounding boxes. The code for this was taken from the official PointRCNN code base https://github.com/sshaoshuai/PointRCNN.
  - **logs**
    - **loss** : This folder stores the training and validation tensorboard files.
  - **pointnet2_ops_lib** : This folder contains the code for farthest point sampling and associated functionalities. The code for this was taken from the official PointRCNN code base https://github.com/sshaoshuai/PointRCNN.
  - **results** : This folder stores the class and the coordinates of the bounding boxes predicted the test dataset.
  - **weights** : This folder stores the training checkpoints
  - dataset.py : This file contains the dataset class which will read the files from the dataset directory and prepare them for training
  - eval_map.py : This file contains the function to calculate the mean average precision(mAP) of the predicted outputs of the network wrt to the ground truths
  - eval_model.py : This file loads the specified checkpoint and outputs the calculated mAP of the model.
  - iou.py : This file contains the functions to calculate the 3D IoU values for the given pairs of bounding boxes
  - loss.py : This file contains the loss calculation functions.
  - PointGCN.py : This file contains the class which defines the network architecture.
  - pred2label.py : This file converts predictions which are stored in .pt format to json files.
  - spatialGraphConv.py : This file contains the class definition for the spatial graph convolution layer.
  - test.py : This file contains the function for validation
  - train.py : This file trains the model.

## Running the code

Before running the codebase, the following 2 commands need to be run to install the iou and pointnet ops modules:
```
cd pointnet2_ops_lib
python setup.py install
cd ../iou3d
python setup.py install
cd ..
```

Once these modules are successfully compiled, to run the training run the command:
```
python train.py
```

In the main function of the  ```train.py``` file, the values for training parameters are initialized and can be changed as per requirements.

Among the defined parameters, there is a variable called ```validation_split``` which will divide the dataset into a training and valiation set. The pointclouds which are part of the validation split will be passed through model and the validation loss calculated, and the predicted bounding boxes and classification scores for every pointcloud in the validation set will be save in the results directory as .pt files.
