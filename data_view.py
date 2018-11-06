# External dependencies
from __future__ import print_function
import os
import sys
import argparse
import logging
import glob
import re
import cv2
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def read_compressed_image(path):
    BIN = os.path.join(os.path.dirname(__file__), "ljpeg", "jpegdir", "jpeg")
    PATTERN = re.compile('\sC:(\d+)\s+N:(\S+)\s+W:(\d+)\s+H:(\d+)\s')
    cmd = '%s -d -s %s' % (BIN, path)
    output = subprocess.check_output(cmd, shell=True)
    m = re.search(PATTERN, output.decode('utf-8'))
    C = int(m.group(1))  # Assumes this is number of channels
    file = m.group(2)
    W = int(m.group(3))
    H = int(m.group(4))
    assert C == 1
    im = np.fromfile(file, dtype='uint16').reshape(H, W)
    L = im >> 8
    H = im & 0xFF
    im = (H << 8) | L
    os.remove(file)
    return im


def read_ics(img_folder):
    ics_path = glob.glob(img_folder + '/*.ics')[0]
    data_dict = {}
    with open(ics_path, "r") as ics_file:
        # find the shape of image
        for line in ics_file.readlines():
            dims_dict = {"H": None, "W": None}
            line = line.strip().split(' ')
            if len(line) < 7:
                continue
            dims_dict["H"] = int(line[2])
            dims_dict["W"] = int(line[4])
            dims_dict["bps"] = int(line[6])
            dims_dict["overlay"] = True if line[-1] == "OVERLAY" else False
            if dims_dict["bps"] != 12:
                logging.warning('Bits per pixel != 12: %s' % line[0])
            data_dict[line[0]] = dims_dict
        for _, v in data_dict.items():
            assert v["H"] is not None
            assert v["W"] is not None
    return data_dict


def image_display(img_file):
    with Image.open(img_file) as img:
        plt.imshow(img)
        plt.show()


def ljpeg_emulator(ljpeg_path, ics_dict, img_format='.jpeg', normalize=True, verify=False, scale=None):
    logging.basicConfig(level=logging.INFO)
    assert 'LJPEG' in ljpeg_path
    stem = os.path.splitext(ljpeg_path)[0]
    name = ljpeg_path.split('.')[-2]
    output_path = stem + img_format
    image = read_compressed_image(ljpeg_path)
    reshape = False
    if ics_dict[name]["W"] != image.shape[1]:
        logging.warning('reshape: %s' % ljpeg_path)
        image = image.reshape((ics_dict[name]["H"], ics_dict[name]["W"]))
        reshape = True
    raw = image
    if normalize:
        logging.warning("normalizing color, will lose information")
        if verify:
            logging.error("verification is going to fail")
        if scale:
            rows, cols = image.shape
            image = cv2.resize(image, (int(cols * scale), int(rows * scale)))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)
    elif scale:
        logging.error("--scale must be used with --visual")
        sys.exit(1)
        # image = cv2.equalizeHist(image)
    cv2.imwrite(output_path, image)
    if verify:
        verify = cv2.imread(output_path, -1)
        if np.all((raw.reshape if reshape else raw) == verify):
            logging.info('Verification successful, conversion is lossless')
        else:
            logging.error('Verification failed: %s' % ljpeg_path)
    # image_display(output_path)


def read_overlay(overlay_path):
    with open(overlay_path, "r") as overlay_file:
        lines = overlay_file.readlines()
    total_abnormalities = int(lines[0][len("TOTAL_ABNORMALITIES"):])
    overlays = []
    line_offset = 0
    for abnormality in range(total_abnormalities):
        abnormality_dict = {}
        lesion_info = lines[line_offset + 2].split(" ")
        info = 0
        while info < len(lesion_info):
            abnormality_dict[lesion_info[info].strip()] = lesion_info[info + 1].strip()
            info += 2
        abnormality_dict["ASSESSMENT"] = int(lines[line_offset + 3][len("ASSESSMENT"):].strip())
        abnormality_dict["SUBTLETY"] = int(lines[line_offset + 4][len("SUBTLETY"):].strip())
        abnormality_dict["PATHOLOGY"] = lines[line_offset + 5][len("PATHOLOGY"):].strip()
        total_outlines = int(lines[line_offset + 6][len("TOTAL_OUTLINES"):].strip())
        abnormality_dict["outlines"] = {}
        for outline in range(1, total_outlines + 1):
            if outline == 1:
                abnormality_dict["outlines"]["BOUNDARY"] = list(map(int,
                                                                    lines[line_offset + 6 + outline*2].split(" ")[:-1]))
            else:
                key = "CORE" + str(outline)
                abnormality_dict["outlines"][key] = list(map(int, lines[line_offset + 6 + outline*2].split(" ")[:-1]))
        overlays.append(abnormality_dict)
        line_offset += 6 + 1 + total_outlines * 2
    return overlays


def read_case():
    parser = argparse.ArgumentParser(description='Visualizing DDSM data')
    parser.add_argument('--image-folder', type=str, default='train', metavar='I',
                        help="Directory where images reside.")
    args = parser.parse_args()
    image_folder = args.image_folder
    ics_dict = read_ics(image_folder)
    for f in os.listdir(image_folder):
        if "LJPEG" in f:
            ljpeg_emulator(os.path.join(image_folder, f), ics_dict)
        if "OVERLAY" in f and ics_dict[f.split(".")[1]]["overlay"]:
            ics_dict[f.split(".")[1]]["overlays"] = read_overlay(os.path.join(image_folder, f))
        else:
            logging.info("Skipping %s" % f)


if __name__ == "__main__":
    read_case()
