"""
Mask R-CNN
Configurations and data loading code for DDSM.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Modified by Yair Schiff

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 ddsm.py train --dataset=/path/to/ddsm/ --model=coco

    # Train a new model starting from ImageNet weights.
    python3 ddsm.py train --dataset=/path/to/ddsm/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 ddsm.py train --dataset=/path/to/ddsm/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 ddsm.py train --dataset=/path/to/ddsm/ --model=last

    # Run DDSM evaluation on the last model you trained
    python3 ddsm.py evaluate --dataset=/path/to/ddsm/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
from imgaug import augmenters as iaa
import skimage.io
import tensorflow as tf
from tensorflow.python.client import device_lib

import pdb

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1 

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Steps per epoch should be number of images / batch size (= images per gpu * number of gpus)
    #STEPS_PER_EPOCH = 3609 / (IMAGES_PER_GPU * GPU_COUNT) 
    # Exclude brightened
    STEPS_PER_EPOCH = 3096 / (IMAGES_PER_GPU * GPU_COUNT) 
    
    # Validation steps is equal to number of validation images / batch size
    #TODO: This needs to change for INBreast data
    #VALIDATION_STEPS = 401 / (IMAGES_PER_GPU * GPU_COUNT)
    # Exclude brightened
    VALIDATION_STEPS = 348 / (IMAGES_PER_GPU * GPU_COUNT)

    # Backbone of the model will be resnet50 as opposed to resnet101, since DDSM has few (3k) training images
    BACKBONE = "resnet50"
    
    # TODO: Figure out if we need to change backbone strides
    #BACKBONE_STRIDES = [2, 4, 8, 16, 32, 64]
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # benigns + malignant

    # Length of square anchor side in pixels: added 2, 4, 8 to try to capture smaller masks
    #RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    #IMAGE_RESIZE_MODE = "crop"
    # Yair: Set resize mode to None since using imgaug
    IMAGE_RESIZE_MODE = "crop" 
    IMAGE_MIN_DIM = (4096, 2048)
    IMAGE_MAX_DIM = (4096, 2048)
    
    # Number of channels: 1 --> since we are using grayscale
    IMAGE_CHANNEL_COUNT = 1
    #IMAGE_CHANNEL_COUNT = 3

    # Mean pixel should be 1 value (as opposed to list of 3) --> since we are using grayscale
    MEAN_PIXEL = np.array([81.535])  # calculated for training set annotations only, non-brightened: 81.53458118658982
    #MEAN_PIXEL = np.array([81.638])  # calculated for validation set annotations only, non-brightened: 81.63792052552432

    # Maximum number of ground truth instances to use in one image changed to 15 since have much fewer masks in our dataset
    MAX_GT_INSTANCES = 6 

    # Max number of final detections changed to 15, same as just above
    DETECTION_MAX_INSTANCES = 6


############################################################
#  Dataset
############################################################

class DDSMDataset(utils.Dataset):
    def load_ddsm(self, dataset_dir, subset, class_ids=None, return_ddsm=False, exclude_brightened=False):
        """Load a subset of the DDSM dataset.
        dataset_dir: The root directory of the DDSM dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        return_ddsm: If True, returns the DDSM object.
        """
        ddsm = COCO("{}/annotations/instances_{}.json".format(dataset_dir, subset))
        image_dir = "{}/{}".format(dataset_dir, subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(ddsm.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(ddsm.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        
        else:
            # All images
            image_ids = list(ddsm.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("ddsm", i, ddsm.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            if exclude_brightened and ddsm.imgs[i]["brightened"]:
                continue
            self.add_image(
                "ddsm", image_id=i,
                case_name=ddsm.imgs[i]["case_name"],
                ddsm_name=ddsm.imgs[i]["ddsm_name"],
                digitizer=ddsm.imgs[i]["digitizer"],
                flipped=ddsm.imgs[i]["flipped"],
                brightened=ddsm.imgs[i]["brightened"],
                path=os.path.join(image_dir, ddsm.imgs[i]['file_name']),
                width=ddsm.imgs[i]["width"],
                height=ddsm.imgs[i]["height"],
                annotations=ddsm.loadAnns(ddsm.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_ddsm:
            return ddsm

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a DDSM image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ddsm":
            return super(DDSMDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "ddsm.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(DDSMDataset, self).load_mask(image_id)

    # Override utils.Dataset load_image method to accommodate grayscale
    def load_image(self, image_id):
        """
        Load the specified image and return a [H,W,1] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'], as_gray=True)
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        return image


    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(DDSMDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  DDSM Evaluation
############################################################

def build_ddsm_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match DDSM specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "ddsm"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_ddsm(model, dataset, ddsm, eval_type="bbox", limit=0, image_ids=None):
    """Runs official DDSM evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick DDSM images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding DDSM image IDs.
    ddsm_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        #pdb.set_trace()
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to DDSM format
        # Cast masks to uint8 because DDSM tools errors out on bool
        image_results = build_ddsm_results(dataset, ddsm_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    ddsm_results = ddsm.loadRes(results)

    # Evaluate
    ddsmEval = COCOeval(ddsm, ddsm_results, eval_type)
    #ddsmEval.params.maxDets = [1, 10]
    #ddsmEval.params.iouThrs - np.linspace(.3, 0.95, np.round((0.95 - .3) / .05) + 1, endpoint=True)
    #ddsmEval.params.imgIds = ddsm_image_ids
    ddsmEval.evaluate()
    ddsmEval.accumulate()
    ddsmEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    return ddsmEval.eval


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
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
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
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

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

    print(device_lib.list_local_devices())
    # Create model
    if args.command == "train":
        with tf.device("/gpu:0"):
            model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
        #model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
        
    else:
        with tf.device("/gpu:0"):
            model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
        #model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    # For stage1:
    #model.load_weights(model_path, by_name=True, exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    #model.load_weights(model_path, by_name=True, exclude=["conv1"])
    # For later stages
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = DDSMDataset()
        dataset_train.load_ddsm(args.dataset, "train", exclude_brightened=True)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = DDSMDataset()
        dataset_val.load_ddsm(args.dataset, "val", exclude_brightened=True)
        dataset_val.prepare()

        # Image Augmentation
        augmentation = iaa.OneOf([
            iaa.Sequential([iaa.Scale({"height": (1.0), "width": (1.0)}), iaa.CropToFixedSize(height=4096, width=2048, position="left-center")]),
            iaa.Sequential([iaa.Scale({"height": (0.5), "width": (0.5)}), iaa.CropToFixedSize(height=2048, width=1024, position="left-center")]),
            iaa.Sequential([iaa.Scale({"height": (0.25), "width": (0.25)}), iaa.CropToFixedSize(height=1024, width=512, position="left-center")])
            ]) 
        
        # Right/Left flip 50% of the time
        #augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        #with tf.device("/gpu:0"):
            #model.train(dataset_train, dataset_val,
                    #learning_rate=config.LEARNING_RATE,
                    #epochs=1,
                    #layers='heads',
                    #augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        with tf.device("/gpu:0"):
            model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=40,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        #print("Fine tune all layers")
        #with tf.device("/gpu:0"):
            #model.train(dataset_train, dataset_val,
                    #learning_rate=config.LEARNING_RATE / 10,
                    #epochs=80,
                    #layers='all')#,augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = DDSMDataset()
        val_type = "val"
        ddsm = dataset_val.load_ddsm(args.dataset, val_type, return_ddsm=True)
        dataset_val.prepare()
        print("Running DDSM evaluation on {} images.".format(args.limit))
        evaluate_ddsm(model, dataset_val, ddsm, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
