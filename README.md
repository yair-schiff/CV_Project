# Computer Vision Fall 2018 Project
**Authors:** Yair Schiff (yzs208@nyu.edu) and Joanna Bitton (jtb470@nyu.edu)

**Instructor:** Prof. Rob Fergus

## Overview:
This repository applies the [Mask R-CNN](https://github.com/matterport/Mask_RCNN) architecture and other Convolutional Neural Network architectures to breast mammography data, namely the "Digital Database for Screening Mammography" (DDSM) and INbreast datasets.


- The [data preparation](https://github.com/yair-schiff/CV_Project/tree/master/data_prep) directory contains scripts for processing the DDSM and INbreast data set for the segmentation and classification tasks.
- The [classification](https://github.com/yair-schiff/CV_Project/tree/master/classification) directory contains scripts for training and evaluating classification of mass presence in a mammography image.
- The [Mask R-CNN](https://github.com/yair-schiff/CV_Project/tree/master/Mask_RCNN) directory contains scripts for running the [matterport implemenation](https://github.com/matterport/Mask_RCNN) of the Mask R-CNN algorithm on the mammography data.
