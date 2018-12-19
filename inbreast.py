# External dependencies
from __future__ import print_function
import argparse
import cv2
import glob
import gzip
import json
import logging
import os
import pandas as pd
import pydicom
import re
import shutil
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import shape


# Global variables
logging.basicConfig(level=logging.INFO)
IMG_ID = -1
ANN_ID = -1


def img_id_increment():
    global IMG_ID
    IMG_ID += 1
    temp = IMG_ID
    return temp


def read_dicoms(cases_folder, data_folder, df, img_format=".jpg", normalize=True):
    dcm_folder = os.path.join(cases_folder, "AllDICOMs")
    for f in os.listdir(dcm_folder):
        if ".dcm" in f:
            file_id = f.split("_")[0]
            img_id = img_id_increment()
            output_file = "{:08d}{}".format(img_id, img_format)
            with pydicom.dcmread(os.path.join(dcm_folder, f)) as dcm_file:
                image = dcm_file.pixel_array
                if normalize:
                    logging.warning("normalizing color, will lose information")
                    image = cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                    image = np.uint8(image)
                    # Flip horizontally if laterality == "R"
                    if df.loc["File Name" == file_id]["Flipped"]:
                        image = cv2.flip(image, 1)
                        logging.warning("Flipping image with file name: {}.".format(f))
                cv2.imwrite(os.path.join(data_folder, output_file), image)  # save image


def read_csv(cases_folder):
    csv_file = os.path.join(cases_folder, "INbreast.csv")
    df = pd.DataFrame.from_csv(csv_file, header=0, sep=";")
    df["flipped"] = [flipped for flipped in df["Laterality"] == "R"]
    return df

def main():
    parser = argparse.ArgumentParser(description='Processing INbreast data to COCO format')
    parser.add_argument('--cases', type=str, default='cases', metavar='C',
                        help="Directory where cases reside.")
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="Directory where data are to be saved.")
    parser.add_argument('--enable-log', type=str, default='y', metavar='L',
                        help="Set flag to \'y\' to enable logger (default) and \'n\' to disable logger.")
    args = parser.parse_args()

    cases_folder = args.cases
    data_folder = args.data
    if args.enable_log == "n":
        logging.disable()

    if not os.path.exists(cases_folder):
        raise RuntimeError("Could not find {}.".format(cases_folder))
    if not os.path.exists(data_folder):
        logging.warning("Could not find {}. Creating new folder and processing data". format(data_folder))
        os.mkdir(data_folder)
    if not os.path.exists(os.path.join(data_folder, "annotations")):
        os.mkdir(os.path.join(data_folder, "annotations"))
    if not os.path.exists(os.path.join(data_folder, "test")):
        os.mkdir(os.path.join(data_folder, "test"))

    df = read_csv(cases_folder)
    read_dicoms(cases_folder, data_folder, df)


if __name__ == "__main__":
    main()

