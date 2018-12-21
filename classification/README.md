# Classification

## Overview
The files in this directory enable the training and evaluation of architectures used to classify the DDSM dataset and evaluate the INbreast dataset.

## Requirements
- `pip`

To install the necessary packages use:
```bash
pip install -r requirements_classify.txt
```

## How to use
### Training
To train the ResNet-18 architecture use the `classification_tumor.py` and `classification.py` scripts. The former will classify images as containing a mass or not and the latter will classify images as `benign`, `malignant`, or `normal`. As the first script was reported on, it is better maintained and recommended for usage.

The classification can also be run using the Jupyter notebook. To use either of the scripts above in `classification.ipynb`, make sure the proper code is imported (i.e. `classification` vs. `classification_tumor`).


### Evaluation
To evaluate the model on INbreast data use the `inbreast_evaluation.py` script, which will classify the images in this test set as having a mass or not. (NOTE: this script is not currently capable of classifying INbreast data into `benign`, `malignant`, and `normal` classes).

The script will evaluate each image in the training set, as well as produce a Precision-Recall curve and the AUC score:
[precision_recall_curve]: https://github.com/yair-schiff/CV_Project/blob/master/classification/resnet_pr_curve.png "Precison/Recall Curve"
