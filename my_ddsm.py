"""
Mask R-CNN
Configurations and data loading code for DDSM.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluation on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import json
import numpy as np

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".

import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

from config import Config
import utils
import model as modellib

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# # Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class DDSMConfig(Config):
    """Configuration for training on DDSM.
    Derives from the base Config class and overrides values specific
    to the DDSM dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ddsm"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes
    NUM_CLASSES = 3  # normal, benign, malignant


############################################################
#  Dataset
############################################################

class DDSMDataset(utils.Dataset):
    def load_ddsm(self, dataset_dir, subset):
        """Load a subset of the DDSM dataset.
        dataset_dir: The root directory of the DDSM dataset.
        subset: What to load (train, val)
        return_ddsm: If True, returns the ddsm object.
        """
        subset_dir = '{}/{}'.format(dataset_dir, subset)
        image_dir = '{}/images'.format(subset_dir)
        annotations_dir = '{}/annotations'.format(subset_dir)

        """
        For this function to work:
        - We need to have an instances json file that tells us metadata
          about each of the train/val/test splits
        - This file will tell us which images to load (filename, height, width)
        - Other option is to walk through subset dir and just find all the files
          in that subset (os.walk)
        - We also load in the annotations as well (seems to want an array? coco.loadAnns)
        """
        class_ids = ['normal', 'benign', 'malignant']

        for i in range(0, len(class_ids)):
            self.add_class("ddsm", i, class_ids[i])

        for filename in os.listdir(annotations_dir):
            with open(os.path.join(annotations_dir, filename), 'r') as file:
                ann_dict = json.loads(file.read())
                self.add_image(
                    "ddsm", image_id=os.path.join(annotations_dir, filename),
                    path=os.path.join(image_dir, f"{ann_dict['images'][0]['file_name']}.jpeg"),
                    width=ann_dict['images'][0]["width"],
                    height=ann_dict['images'][0]["height"],
                    annotations=ann_dict['images'][0]["annotations"])

    def load_mask(self, annotation_path):
        """Load instance masks for the given image.

        load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances]

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        ann_dict = None
        with open(annotation_path, 'r') as file:
            ann_dict = json.loads(file.read())

        if ann_dict is None:
            return super(DDSMDataset, self).load_mask(None)

        instance_masks = []
        class_ids = []
        for ann in ann_dict['annotations']:
            mask = np.zeros((ann_dict['images'][0]['height'], ann_dict['images'][0]['width']))
            np_verts = np.array([ann['border']], dtype=np.int32)
            cv2.fillPoly(mask, np_verts, 1)
            instance_masks.append(mask)
            class_ids.append(ann['category_id'])

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(DDSMDataset, self).load_mask(None)


############################################################
#  DDSM Evaluation
############################################################

def build_ddsm_results(dataset, ann_paths, rois, class_ids, scores, masks):
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for ann_path in ann_paths:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "ann_path": ann_path,
                "category_id": dataset.get_source_class_id(class_id, "ddsm"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": mask
            }
            results.append(result)
    return results


def evaluate_ddsm(model, dataset, eval_type="bbox", limit=0):
    """Runs official evaluation.
    dataset: A Dataset object with validation data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    pass
    # # Pick images from the dataset
    # image_ids = image_ids or dataset.image_ids
    #
    # # Limit to a subset
    # if limit:
    #     image_ids = image_ids[:limit]
    #
    # # Get corresponding image IDs.
    # ddsm_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
    #
    # t_prediction = 0
    # t_start = time.time()
    #
    # results = []
    # for i, image_id in enumerate(image_ids):
    #     # Load image
    #     image = dataset.load_image(image_id)
    #
    #     # Run detection
    #     t = time.time()
    #     r = model.detect([image])[0]
    #     t_prediction += (time.time() - t)
    #
    #     # Convert results to COCO format
    #     image_results = build_ddsm_results(dataset, ddsm_image_ids[i:i + 1],
    #                                        r["rois"], r["class_ids"],
    #                                        r["scores"], r["masks"])
    #     results.extend(image_results)
    #
    # """
    # Will have to create our own official evaluation method. This specifically relies on the images
    # being from the COCO dataset (99% sure)
    # """
    # # Load results. This modifies results with additional attributes.
    # coco_results = coco.loadRes(results)
    #
    # # Evaluate
    # cocoEval = COCOeval(coco, coco_results, eval_type)
    # cocoEval.params.imgIds = coco_image_ids
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    #
    # print("Prediction time: {}. Average {}/image".format(
    #     t_prediction, t_prediction / len(image_ids)))
    # print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on DDSM.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on DDSM")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/ddsm/",
                        help='Directory of the DDSM dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'ddsm'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Dataset: ", args.dataset)
    # print("Model: ", args.model)
    print("Logs: ", args.logs)
    print("Limit: ", args.limit)

    # Configurations
    if args.command == "train":
        config = DDSMConfig()
    else:
        class InferenceConfig(DDSMConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    """
    Since we set the config to be DDSMConfig, it will change the final amount
    of classes to be what we specified in the config: 3 (background, benign, malignant)
    """
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    """
    For now, we don't have a .pth file, so I've commented out all of the code for that
    
    Once we have a .pth file, we can uncomment this stuff
    
    However we can pass in the imagenet weights if we don't wanna train from the beginning
    """
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = DDSMDataset()
        dataset_train.load_ddsm(args.dataset, "train")
        # dataset_train.load_ddsm(args.dataset, "val")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = DDSMDataset()
        dataset_val.load_ddsm(args.dataset, "val")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train_model(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=120,
        #             layers='4+')
        #
        # # Training - Stage 3
        # # Fine tune all layers
        # print("Fine tune all layers")
        # model.train_model(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=160,
        #             layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = DDSMDataset()
        dataset_val.load_ddsm(args.dataset, "minival")
        dataset_val.prepare()
        print("Running DDSM evaluation on {} images.".format(args.limit))
        evaluate_ddsm(model, dataset_val, "bbox", limit=int(args.limit))
        evaluate_ddsm(model, dataset_val, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
