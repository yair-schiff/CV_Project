# External dependencies
from __future__ import print_function

import argparse
import logging
import os

import cv2
import numpy as np
import pandas as pd
import pydicom

# import matplotlib.pyplot as plt

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
    file_ids = []
    image_ids = []
    for f in os.listdir(dcm_folder):
        if ".dcm" in f:
            file_id = f.split("_")[0]
            img_id = img_id_increment()
            output_file = "{:08d}{}".format(img_id, img_format)
            file_ids.append(file_id)
            image_ids.append(output_file)
            with pydicom.dcmread(os.path.join(dcm_folder, f)) as dcm_file:
                image = dcm_file.pixel_array
                if normalize:
                    logging.warning("normalizing color, will lose information")
                    image = cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                    image = np.uint8(image)
                    # Flip horizontally if laterality == "R"
                    if df.loc[df["File Name"] == int(file_id)]["flipped"].values[0]:
                        image = cv2.flip(image, 1)
                        logging.warning("Flipping image with file name: {}.".format(f))
                    cv2.imwrite(os.path.join(data_folder, "test", output_file), image)  # save image
    df_file_ids = pd.DataFrame({"file_ids": file_ids, "image_ids": image_ids})
    df_file_ids.to_csv(os.path.join(cases_folder, "INbreast_file_to_id.csv"), index=False)


def read_csv(cases_folder):
    csv_file = os.path.join(cases_folder, "INbreast.csv")
    df = pd.DataFrame.from_csv(csv_file, header=0, sep=";")
    df["flipped"] = [flipped for flipped in df["Laterality"] == "R"]
    return df


def main():
    parser = argparse.ArgumentParser(description='Processing INbreast data to COCO format')
    parser.add_argument('--cases', type=str, default='INbreast/cases', metavar='C',
                        help="Directory where cases reside.")
    parser.add_argument('--data', type=str, default='INbreast/data', metavar='D',
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
    if not os.path.exists(os.path.join(data_folder, "test")):
        os.mkdir(os.path.join(data_folder, "test"))

    df = read_csv(cases_folder)
    read_dicoms(cases_folder, data_folder, df)


if __name__ == "__main__":
    main()
