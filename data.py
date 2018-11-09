# External dependencies
from __future__ import print_function
import argparse
import cv2
import glob
import json
import logging
import os
import re
import subprocess
import sys
import numpy as np

# Global variables
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data-prep-logger')
# logger.setLevel(level=logging.INFO)
IMG_ID = 0
ANN_ID = 0


def img_id_increment():
    global IMG_ID
    IMG_ID += 1
    temp = IMG_ID
    return temp


def ann_id_increment():
    global ANN_ID
    ANN_ID += 1
    temp = ANN_ID
    return temp


def read_ics(case_folder):
    ics_path = glob.glob(case_folder + '/*.ics')[0]
    data_dict = {}
    with open(ics_path, "r") as ics_file:
        for line in ics_file.readlines():
            dims_dict = {"H": None, "W": None}
            line = line.strip().split(" ")
            if len(line) < 7:
                if "ics_version" in line:
                    data_dict["version"] = line[0] + line[1]
                elif "PATIENT_AGE" in line:
                    data_dict["patient_age"] = line[1]
                elif "DATE_DIGITIZED" in line:
                    year = line[-1]
                    month = ("0" + line[-3]) if len(line[-3]) == 1 else line[-3]
                    day = ("0" + line[-2]) if len(line[-2]) == 1 else line[-2]
                    data_dict["date"] = year + month + day
                continue
            dims_dict["H"] = int(line[2])
            dims_dict["W"] = int(line[4])
            dims_dict["bps"] = int(line[6])
            dims_dict["overlay"] = True if line[-1] == "OVERLAY" else False
            if dims_dict["bps"] != 12:
                logger.warning('Bits per pixel != 12: %s' % line[0])
            data_dict[line[0]] = dims_dict
        for k, v in data_dict.items():
            if k != "version" and k != "patient_age" and k != "date":
                assert v["H"] is not None
                assert v["W"] is not None
    return data_dict


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


def ljpeg_emulator(ljpeg_path, ics_dict, data_folder, img_format='.jpg', normalize=True, verify=False, scale=None):
    assert 'LJPEG' in ljpeg_path
    stem = os.path.splitext(ljpeg_path)[0]
    name = ljpeg_path.split('.')[-2]
    output_file = stem.split("/")[-1] + img_format
    image = read_compressed_image(ljpeg_path)
    reshape = False
    if ics_dict[name]["W"] != image.shape[1]:
        logger.warning('reshape: %s' % ljpeg_path)
        image = image.reshape((ics_dict[name]["H"], ics_dict[name]["W"]))
        reshape = True
    raw = image
    if normalize:
        logger.warning("normalizing color, will lose information")
        if verify:
            logger.error("verification is going to fail")
        if scale:
            rows, cols = image.shape
            image = cv2.resize(image, (int(cols * scale), int(rows * scale)))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)
    elif scale:
        logger.error("--scale must be used with --visual")
        sys.exit(1)
        # image = cv2.equalizeHist(image)
    if "RIGHT" in ljpeg_path:
        image = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(data_folder, output_file), image)  # save image
    if verify:
        verify = cv2.imread(output_file, -1)
        if np.all((raw.reshape if reshape else raw) == verify):
            logger.info('Verification successful, conversion is lossless')
        else:
            logger.error('Verification failed: %s' % ljpeg_path)


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
        outlines = []
        for outline in range(1, total_outlines + 1):
            out = list(map(int, lines[line_offset + 6 + outline*2].split(" ")[:-1]))
            outlines.append(out)

        abnormality_dict["outlines"] = outlines
        overlays.append(abnormality_dict)
        line_offset += 6 + 1 + total_outlines * 2
    return overlays


def apply_chain_code(code, pixel):
    if code == 0:
        return pixel[0], pixel[1] - 1
    elif code == 1:
        return pixel[0] + 1, pixel[1] - 1
    elif code == 2:
        return pixel[0] + 1, pixel[1]
    elif code == 3:
        return pixel[0] + 1, pixel[1] + 1
    elif code == 4:
        return pixel[0], pixel[1] + 1
    elif code == 5:
        return pixel[0] - 1, pixel[1] + 1
    elif code == 6:
        return pixel[0] - 1, pixel[1]
    elif code == 7:
        return pixel[0] - 1, pixel[1] - 1


def get_polygon(chain_code):
    tup_verts = []
    current_pixel = (chain_code[0], chain_code[1])  # pixels are of form x,y (i.e. column, row)
    tup_verts.append((chain_code[0], chain_code[1]))
    for c in range(2, len(chain_code)):
        current_pixel = apply_chain_code(chain_code[c], current_pixel)
        tup_verts.append((current_pixel[0], current_pixel[1]))
    return tup_verts


def flip_polygon(tup_verts, width):
    midline = width // 2
    flipped = []
    for i in range(len(tup_verts)):
        dist_to_mid = midline - tup_verts[i][0]
        new_vert = midline + dist_to_mid, tup_verts[i][1]
        flipped.append(new_vert)
    return flipped


def get_bbox(tup_verts):
    bottom_border = -1
    top_border = 1000000
    right_border = -1
    left_border = 1000000
    for vertex in tup_verts:
        bottom_border = max(bottom_border, vertex[1])
        top_border = min(top_border, vertex[1])
        right_border = max(right_border, vertex[0])
        left_border = min(left_border, vertex[0])
    height = bottom_border - top_border
    width = right_border - left_border
    return [left_border, top_border, width, height]


# TODO: Move this code to somewhere else like ddsm.py or model.py
def get_mask(tup_verts, cat_id, height, width):
    mask = np.zeros((height, width, 2))
    for vert in tup_verts:
        mask[vert[1], vert[0], cat_id - 1] = 1
    return mask


def get_cat(pathology):
    if "normal" in pathology:
        return 0, "normal"
    if "BENIGN" in pathology:
        return 1, "benign"
    if "MALIGNANT" in pathology:
        return 2, "malignant"


def create_json(json_name, img, ics_info, data_folder):
    info = {
        "version": ics_info["version"],
        "description": "Digital Database for Screening Mammography (DDSM)",
        "url": "http://marathon.csee.usf.edu/Mammography/Database.html",
        "date_created": ics_info["date"],
        "patient_age": ics_info["patient_age"]
    }
    images = [{
        "id": img_id_increment(),
        "width": ics_info[img]["W"],
        "height": ics_info[img]["H"],
        "file_name": json_name,
        "license": 1
    }]
    # licenses = [{
    #     "id": 1,
    #     "name": "The MIT License (MIT)",
    #     "url": "https://opensource.org/licenses/MIT"}
    # ]
    categories = [
        {
            'id': 1,
            'name': 'normal',
            'supercategory': 'normal',
        },
        {
            'id': 2,
            'name': 'benign',
            'supercategory': 'tumor',
        },
        {
            'id': 3,
            'name': 'malignant',
            'supercategory': 'tumor',
        },
    ]

    annotations = []
    if ics_info[img]["overlay"]:
        for overlay in ics_info[img]["overlays"]:
            category_id = get_cat(overlay["PATHOLOGY"])[0]
            birads_id = overlay["ASSESSMENT"]
            subtlety_id = overlay["SUBTLETY"]
            for outline in overlay["outlines"]:
                border = get_polygon(outline)
                if "RIGHT" in json_name:
                    border = flip_polygon(border, ics_info[img]["W"])
                bbox = get_bbox(border)
                # segmentation = get_mask(border, category_id, ics_info[img]["H"], ics_info[img]["W"]).tolist()
                annotation = {
                    "id": ann_id_increment(),
                    "image_id": IMG_ID,
                    "category_id": category_id,
                    "birads_id": birads_id,
                    "subtlety_id": subtlety_id,
                    # "segmentation": segmentation,
                    "border": border,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                annotations.append(annotation)
    img_dict = {
        "info": info,
        "images": images,
        "annotations": annotations,
        # "license" = licenses,
        "categories": categories
    }

    json_file = os.path.join(data_folder, json_name + ".json")
    with open(json_file, 'w') as fp:
        json.dump(img_dict, fp)


def read_case(case_folder, data_folder):
    ics_dict = read_ics(case_folder)
    img_prefix = ""
    for f in os.listdir(case_folder):
        if "LJPEG" in f:
            img_prefix = f.split(".")[0]
            ljpeg_emulator(os.path.join(case_folder, f), ics_dict, os.path.join(data_folder, "images"))
        elif "OVERLAY" in f and ics_dict[f.split(".")[1]]["overlay"]:
            ics_dict[f.split(".")[1]]["overlays"] = read_overlay(os.path.join(case_folder, f))
        else:
            logger.info("Skipping %s" % f)

    for instance in ics_dict:
        if instance != "version" and instance != "patient_age" and instance != "date":
            json_name = img_prefix + "." + instance
            create_json(json_name, instance, ics_dict, os.path.join(data_folder, "annotations"))


def main():
    parser = argparse.ArgumentParser(description='Processing DDSM data to COCO format')
    parser.add_argument('--cases', type=str, default='cases', metavar='C',
                        help="Directory where cases reside.")
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="Directory where data are to be saved.")
    parser.add_argument('--enable-log', type=str, default='y', metavar='L',
                        help="Set flag to \'y\' to enable logger (default) and \'n\' to disable logger.")
    args = parser.parse_args()

    ROOT_DIR = os.getcwd()
    cases_folder = os.path.join(ROOT_DIR, args.cases)
    data_folder = os.path.join(ROOT_DIR, args.data)

    logger.propagate = args.enable_log == "n"
    logger.disabled = args.enable_log == "n"

    if not os.path.exists(cases_folder):
        raise (RuntimeError("Could not find " + cases_folder +
                            ', please download data from ftp://figment.csee.usf.edu/pub/DDSM/'))

    if not os.path.exists(data_folder):
        logger.warning("Could not find " + data_folder + ". Creating new folder and processing data")
        os.mkdir(data_folder)
        os.mkdir(os.path.join(data_folder, "train"))
        os.mkdir(os.path.join(data_folder, "train/images"))
        os.mkdir(os.path.join(data_folder, "train/annotations"))
    for category in os.listdir(cases_folder):  # walk down categories (benign, malignant, normal)
        if os.path.isdir(os.path.join(cases_folder, category)):  # skip .DS_Store and other non-directory files
            for case in os.listdir(os.path.join(cases_folder, category)):  # walk down cases
                if os.path.isdir(os.path.join(cases_folder, category, case)):
                    for case_folder in os.listdir(os.path.join(cases_folder, category, case)):
                        if os.path.isdir(os.path.join(cases_folder, category, case, case_folder)):
                            read_case(os.path.join(cases_folder, category, case, case_folder),
                                      os.path.join(data_folder, "train"))
    # else:
    #     logger.info(data_folder + " already exists. Data processing assumed to have occurred already.")


if __name__ == "__main__":
    main()
