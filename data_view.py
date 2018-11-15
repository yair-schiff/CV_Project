# External dependencies
from __future__ import print_function
import argparse
import cv2
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


# def get_mask(tup_verts, height, width, classification):
#     label = int(classification) - 1
#     mask = np.zeros((height, width))
#     np_verts = np.array([tup_verts], dtype=np.int32)
#     cv2.fillPoly(mask, np_verts, 1)
#     # mask_for_model = np.zeros((height, width, 2), dtype=np.uint8)
#     # f = open("mask.txt", "w")
#     # for row in range(height):
#     #     for col in range(width):
#     #         mask_for_model[row, col, label] = mask[row, col]
#     #         # f.write("[" + str(mask_for_model[row, col, 0]) + ", " + str(mask_for_model[row, col, 1]) + ", "
#     #         #             + str(mask_for_model[row, col, 2]) + "]")
#     #     # f.write("\n")
#     # f.close()
#     return mask


def image_display(img_file, json_file):
    with open(json_file, "r") as json_fp:
        json_str = json_fp.read()
    json_dict = json.loads(json_str)
    extent = 0, json_dict["images"][0]["width"], 0, json_dict["images"][0]["height"]
    fig, ax = plt.subplots(1)
    with Image.open(img_file) as img:
        ax.imshow(img, cmap=plt.cm.gray, interpolation='none', extent=extent)
    for ann in json_dict["annotations"]:
        ax.add_patch(create_box(ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]))
        ax.add_patch(create_border(ann["border"], fill=True, alpha=0.3))
        ax.annotate(json_dict["categories"][ann["category_id"]]["name"],
                    (ann["bbox"][0] - 50, ann["bbox"][1]+ann["bbox"][3]))
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualizing DDSM data')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help='Data folder')
    parser.add_argument('--image', type=str, default='C_0029_1.LEFT_CC', metavar='I',
                        help="Image name.")
    args = parser.parse_args()
    data_folder = args.data
    image = args.image
    image_display(os.path.join(data_folder, "train/images", image + ".jpg"),
                  os.path.join(data_folder, "train/annotations", image + ".json"))


if __name__ == "__main__":
    main()
