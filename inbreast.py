# External dependencies
from __future__ import print_function
import argparse
import cv2
import glob
import gzip
import json
import logging
import os
import pydicom
import re
import shutil
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import shape


def read_dicoms(cases_folder, data_folder):
    dcm_folder = os.path.join(cases_folder, "AllDICOMs")
    for f in os.listdir(dcm_folder):
        if ".dcm" in f:
            with pydicom.dcmread(os.path.join(dcm_folder, f)) as dcm_file:
                image = dcm_file.pixel_array
                print(np.mean(image, axis=1))
                plt.imshow(image, cmap="gray")
                plt.show()


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
    read_dicoms(cases_folder)


if __name__ == "__main__":
    main()

