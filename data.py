# External dependencies
from __future__ import print_function
import argparse
import cv2
import glob
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import numpy as np

# Global variables
logging.basicConfig(level=logging.INFO)
IMG_ID = -1
ANN_ID = -1


def img_id_increment():
    global IMG_ID
    IMG_ID += 1
    temp = IMG_ID
    return temp


def img_id_decrement():
    global IMG_ID
    IMG_ID -= 1
    temp = IMG_ID
    return temp


def ann_id_increment():
    global ANN_ID
    ANN_ID += 1
    temp = ANN_ID
    return temp


def read_ics(case_folder):
    ics_path = (glob.glob(case_folder + '/*.ics.gz') or glob.glob(case_folder + '/*.ics'))[0]
    data_dict = {}
    if ".gz" in ics_path:
        with gzip.open(ics_path, "rt") as content:
            lines = content.readlines()
    else:
        with open(ics_path, "r") as ics_file:
            lines = ics_file.readlines()
    for line in lines:
        dims_dict = {"H": None, "W": None}
        line = line.strip().split(" ")
        if len(line) < 7:
            if "ics_version" in line:
                data_dict["version"] = line[0] + line[1]
            elif "PATIENT_AGE" in line:
                try:
                    data_dict["patient_age"] = line[1]
                except IndexError:
                    data_dict["patient_age"] = -99
            elif "DATE_DIGITIZED" in line:
                year = line[-1]
                month = ("0" + line[-3]) if len(line[-3]) == 1 else line[-3]
                day = ("0" + line[-2]) if len(line[-2]) == 1 else line[-2]
                data_dict["date"] = "{}-{}-{}".format(year, month, day)
            continue
        dims_dict["H"] = int(line[2])
        dims_dict["W"] = int(line[4])
        dims_dict["bps"] = int(line[6])
        dims_dict["overlay"] = True if line[-1] == "OVERLAY" else False
        if dims_dict["bps"] != 12:
            logging.warning('Bits per pixel != 12: %s' % line[0])
        data_dict[line[0]] = dims_dict
    for k, v in data_dict.items():
        if k != "version" and k != "patient_age" and k != "date":
            assert v["H"] is not None
            assert v["W"] is not None
    return data_dict


def read_compressed_image(path):
    if ".gz" in path:
        with gzip.open(path, "rb") as f_in:
            with open(path[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        path = path[:-3]
    BIN = os.path.join(os.path.dirname(__file__), "ljpeg", "jpegdir", "jpeg")
    PATTERN = re.compile('\sC:(\d+)\s+N:(\S+)\s+W:(\d+)\s+H:(\d+)\s')
    cmd = '%s -d -s %s' % (BIN, path)
    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        logging.warning("Reading of compressed image returned error. "
                        "Will need to delete this image from corpus:{}".format(path))
        return None
    m = re.search(PATTERN, output.decode('utf-8'))
    C = int(m.group(1))  # Assumes this is number of channels
    file = m.group(2)
    W = int(m.group(3))
    H = int(m.group(4))
    assert C == 1
    im = np.fromfile(file, dtype="uint16").reshape(H, W)
    L = im >> 8
    H = im & 0xFF
    im = (H << 8) | L
    os.remove(file)
    return im


def ljpeg_emulator(ljpeg_path, ics_dict, data_folder, img_format='.jpg', normalize=True, verify=False, scale=None):
    assert "LJPEG" in ljpeg_path
    name = ljpeg_path.split(".")[-3] if ".gz" in ljpeg_path else ljpeg_path.split(".")[-2]
    img_id = img_id_increment()
    output_file = "{:08d}{}".format(img_id, img_format)
    image = read_compressed_image(ljpeg_path)
    if image is None:
        img_id_decrement()
        del ics_dict[name]
        return -1
    reshape = False
    if ics_dict[name]["W"] != image.shape[1]:
        logging.warning('\treshape: %s' % ljpeg_path)
        image = image.reshape((ics_dict[name]["H"], ics_dict[name]["W"]))
        reshape = True
    raw = image
    if normalize:
        logging.warning("\tnormalizing color, will lose information")
        if verify:
            logging.error("\tverification is going to fail")
        if scale:
            rows, cols = image.shape
            image = cv2.resize(image, (int(cols * scale), int(rows * scale)))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)
    elif scale:
        logging.error("\t--scale must be used with --visual")
        sys.exit(1)
        # image = cv2.equalizeHist(image)
    if "RIGHT" in ljpeg_path:
        image = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(data_folder, output_file), image)  # save image
    if verify:
        verify = cv2.imread(output_file, -1)
        if np.all((raw.reshape if reshape else raw) == verify):
            logging.info('\tVerification successful, conversion is lossless')
        else:
            logging.error('\tVerification failed: %s' % ljpeg_path)
    return img_id


def read_overlay(overlay_path):
    if ".gz" in overlay_path:
        with gzip.open(overlay_path, "rt") as content:
            lines = content.readlines()
    else:
        with open(overlay_path, "r") as overlay_file:
            lines = overlay_file.readlines()
    total_abnormalities = int(lines[0][len("TOTAL_ABNORMALITIES"):])
    overlays = []
    line_offset = 0
    offsets = {
        "LESION": 2,
        "ASSESSMENT": 3,
        "SUBTLETY": 4,
        "PATHOLOGY": 5,
        "OUTLINES": 6,
        "TOTAL": 6
    }
    reset_offsets = False
    for abnormality in range(total_abnormalities):
        abnormality_dict = {}
        lesion_info = lines[line_offset + offsets["LESION"]].split(" ")
        info = 0
        while info < len(lesion_info):
            try:
                abnormality_dict[lesion_info[info].strip()] = lesion_info[info + 1].strip()
                info += 2
            except IndexError:
                info += 2
                continue
        try:
            abnormality_dict["ASSESSMENT"] = int(lines[line_offset + offsets["ASSESSMENT"]][len("ASSESSMENT"):].strip())
        except ValueError:
            # offsets are off by one, need to increment
            reset_offsets = True
            for key in offsets:
                offsets[key] += 1
            abnormality_dict["ASSESSMENT"] = int(lines[line_offset + offsets["ASSESSMENT"]][len("ASSESSMENT"):].strip())
        abnormality_dict["SUBTLETY"] = int(lines[line_offset + offsets["SUBTLETY"]][len("SUBTLETY"):].strip())
        abnormality_dict["PATHOLOGY"] = lines[line_offset + offsets["PATHOLOGY"]][len("PATHOLOGY"):].strip()
        if get_cat(abnormality_dict["PATHOLOGY"]) is None:
            abnormality_dict["PATHOLOGY"] = overlay_path.split("/")[-4][:-1].upper()
        total_outlines = int(lines[line_offset + offsets["OUTLINES"]][len("TOTAL_OUTLINES"):].strip())
        abnormality_dict["outlines"] = {}
        outlines = []
        for outline in range(1, total_outlines + 1):
            try:
                out = list(map(int, list(filter((lambda x: x != ""), lines[line_offset + offsets["OUTLINES"]
                                                                           + outline*2].strip().split(" ")[:-1]))))
                outlines.append(out)
            except (IndexError, ValueError):
                total_outlines -= 1
                continue
        abnormality_dict["outlines"] = outlines
        overlays.append(abnormality_dict)
        line_offset += offsets["TOTAL"] + total_outlines * 2
        if reset_offsets:
            reset_offsets = False
            for key in offsets:
                offsets[key] -= 1
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


def get_cat(pathology):
    if "normal" in pathology:
        return 0, "normal"
    if "BENIGN" in pathology:
        return 1, "benign"
    if "MALIGNANT" in pathology:
        return 2, "malignant"


def create_image_json(img_id, img, ddsm_file_name, ics_info, img_format=".jpg"):
    file_name = "{:08d}{}".format(img_id, img_format)
    image_json = {
        "license": 1,
        "file_name": file_name,
        "ddsm_name": ddsm_file_name,
        "height": ics_info[img]["H"],
        "width": ics_info[img]["W"],
        "date_captured": ics_info["date"],
        "id": img_id,
        "patient_age": ics_info["patient_age"]
    }
    return image_json


def create_annotation_json(img, ics_info, flip=False):
    annotations = []
    img_id = ics_info[img]["id"]
    if ics_info[img]["overlay"]:
        for overlay in ics_info[img]["overlays"]:
            try:
                category_id = get_cat(overlay["PATHOLOGY"])[0]
            except TypeError:
                continue
            birads_id = overlay["ASSESSMENT"]
            subtlety_id = overlay["SUBTLETY"]
            for outline in overlay["outlines"]:
                border = get_polygon(outline)
                if flip:
                    border = flip_polygon(border, ics_info[img]["W"])
                bbox = get_bbox(border)
                annotation = {
                    "id": ann_id_increment(),
                    "image_id": img_id,
                    "category_id": category_id,
                    "birads_id": birads_id,
                    "subtlety_id": subtlety_id,
                    "segmentation": border,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                annotations.append(annotation)
    else:
        annotation = {
            "id": ann_id_increment(),
            "image_id": img_id,
            "category_id": 0,
            "segmentation": [],
            "bbox": []
            # "bbox": [0, 0, ics_info[img]["W"], ics_info[img]["H"]],
            # "iscrowd": 1
        }
        annotations.append(annotation)
    return annotations


def create_instances_json(images, annotations, data_folder, dataset):
    info = {
        "description": "Digital Database for Screening Mammography (DDSM)",
        "url": "http://marathon.csee.usf.edu/Mammography/Database.html",
        "version": "1.0",
        "year": 2018,
        "contributor": "yair-schiff",
        "date_created": "2018/11/15"
    }
    licenses = [
        {
            "id": 1,
            "name": "Dummy license",
            "url": "dummy_license_url"
        }
    ]
    categories = [
        {
            'id': 0,
            'name': 'normal',
            'supercategory': 'normal',
        },
        {
            'id': 1,
            'name': 'benign',
            'supercategory': 'tumor',
        },
        {
            'id': 2,
            'name': 'malignant',
            'supercategory': 'tumor',
        },
    ]
    instances_json = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    json_file = os.path.join(data_folder, "instances_{}.json".format(dataset))
    with open(json_file, 'w') as fp:
        json.dump(instances_json, fp)


def read_case(case_folder, data_folder):
    logging.info("Working on case: {} ({})".format("/".join(case_folder.split("/")[-3:]), data_folder.split("/")[-1]))
    ics_dict = read_ics(case_folder)
    images = []
    annotations = []
    for f in os.listdir(case_folder):
        if f.split(".")[-1] == "1":  # skip images that end in .1
            continue
        if "LJPEG" in f:
            logging.info("File: {}".format(f))
            img_id = ljpeg_emulator(os.path.join(case_folder, f), ics_dict, os.path.join(data_folder, "images"))
            if img_id == -1:
                continue
            f_split = f.split(".")
            ddsm_file_name = "{}.{}".format(f_split[0], f_split[1])
            images.append(create_image_json(img_id, f_split[1], ddsm_file_name, ics_dict))
            ics_dict[f_split[1]]["id"] = img_id
        elif "OVERLAY" in f and ics_dict[f.split(".")[1]]["overlay"]:
            logging.info("File: {}".format(f))
            ics_dict[f.split(".")[1]]["overlays"] = read_overlay(os.path.join(case_folder, f))
        else:
            logging.warning("Skipping %s" % f)

    for instance in ics_dict:
        if instance != "version" and instance != "patient_age" and instance != "date":
            annotations += create_annotation_json(instance, ics_dict, "RIGHT" in instance)
    return images, annotations


def main():
    parser = argparse.ArgumentParser(description='Processing DDSM data to COCO format')
    parser.add_argument('--cases', type=str, default='cases', metavar='C',
                        help="Directory where cases reside.")
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="Directory where data are to be saved.")
    parser.add_argument('--split', type=int, default=10, metavar='S',
                        help="Number on which to split training/validation sets, i.e. every nth case will be moved to"
                             "validation set. (Default: 10 to create 90/10 split between training and validation sets)")
    parser.add_argument('--enable-log', type=str, default='y', metavar='L',
                        help="Set flag to \'y\' to enable logger (default) and \'n\' to disable logger.")
    args = parser.parse_args()

    ROOT_DIR = os.getcwd()
    cases_folder = os.path.join(ROOT_DIR, args.cases)
    data_folder = os.path.join(ROOT_DIR, args.data)
    data_split = args.split
    if args.enable_log == "n":
        logging.disable()

    images_train = []
    annotations_train = []
    images_val = []
    annotations_val = []
    train_counter = 0  # keeps track of number of cases processed; every 10th case will be moved to validation set
    if not os.path.exists(cases_folder):
        raise (RuntimeError("Could not find " + cases_folder +
                            ', please download data from ftp://figment.csee.usf.edu/pub/DDSM/'))

    if not os.path.exists(data_folder):
        logging.warning("Could not find " + data_folder + ". Creating new folder and processing data")
        os.mkdir(data_folder)
    if not os.path.exists(os.path.join(data_folder, "annotations")):
        os.mkdir(os.path.join(data_folder, "annotations"))
    if not os.path.exists(os.path.join(data_folder, "train")):
        os.mkdir(os.path.join(data_folder, "train"))
    if not os.path.exists(os.path.join(data_folder, "train/images")):
        os.mkdir(os.path.join(data_folder, "train/images"))
    if not os.path.exists(os.path.join(data_folder, "val")):
        os.mkdir(os.path.join(data_folder, "val"))
    if not os.path.exists(os.path.join(data_folder, "val/images")):
        os.mkdir(os.path.join(data_folder, "val/images"))
    for category in os.listdir(cases_folder):  # walk down categories (benign, malignant, normal)
        if os.path.isdir(os.path.join(cases_folder, category)):  # skip .DS_Store and other non-directory files
            for case in os.listdir(os.path.join(cases_folder, category)):  # walk down cases
                if os.path.isdir(os.path.join(cases_folder, category, case)):
                    for case_folder in os.listdir(os.path.join(cases_folder, category, case)):
                        if os.path.isdir(os.path.join(cases_folder, category, case, case_folder)):
                            train_counter += 1
                            if train_counter % data_split == 0:  # make this a validation case
                                ims, anns = read_case(os.path.join(cases_folder, category, case, case_folder),
                                                      os.path.join(data_folder, "val"))
                                images_val += ims
                                annotations_val += anns
                            else:
                                ims, anns = read_case(os.path.join(cases_folder, category, case, case_folder),
                                                      os.path.join(data_folder, "train"))
                                images_train += ims
                                annotations_train += anns
    # Create .json for val and training sets
    create_instances_json(images_train, annotations_train, os.path.join(data_folder, "annotations"), "train")
    create_instances_json(images_val, annotations_val, os.path.join(data_folder, "annotations"), "val")


if __name__ == "__main__":
    main()
