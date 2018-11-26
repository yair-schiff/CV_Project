from __future__ import print_function
import argparse
import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_box(x, y, width, height):
    return patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none')


def create_border(tup_verts, fill=False, alpha=1):
    return patches.Polygon(np.array(tup_verts), closed=False, fill=fill, alpha=alpha, color='r')


def image_display(img_id, img_file, json_file):
    with open(json_file, "r") as json_fp:
        json_str = json_fp.read()
    json_dict = json.loads(json_str)
    print("Visualizing: {}".format(json_dict["images"][img_id]["ddsm_name"]))
    annotations = []
    for ann in json_dict["annotations"]:
        if ann["image_id"] == img_id:
            annotations.append(ann)
    extent = 0, json_dict["images"][img_id-1]["width"], 0, json_dict["images"][img_id-1]["height"]
    fig, ax = plt.subplots(1)
    with Image.open(img_file) as img:
        ax.imshow(img, cmap=plt.cm.gray, interpolation='none', extent=extent)
    print("Image has {} annotations.".format(len(annotations)))
    for ann in annotations:
        if ann["bbox"]:
            ax.add_patch(create_box(ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]))
        if ann["segmentation"]:
            ax.add_patch(create_border(ann["segmentation"], fill=True, alpha=0.3))
        ax.annotate(json_dict["categories"][ann["category_id"]]["name"],
                    (ann["bbox"][0] - 50, ann["bbox"][1]+ann["bbox"][3]))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualizing DDSM data')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help='Data folder')
    parser.add_argument('--image-id', type=str, default='1', metavar='I',
                        help="Image name.")
    args = parser.parse_args()
    data_folder = args.data
    image_id = int(args.image_id)
    image_file = "{:08d}.jpg".format(image_id)
    image_display(image_id, os.path.join(data_folder, "train/images", image_file),
                  os.path.join(data_folder, "annotations", "instances_train.json"))


if __name__ == "__main__":
    main()
