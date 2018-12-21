# Mask R-CNN

## Overview
The majority of code in this directory is from: https://github.com/matterport/Mask_RCNN

The files in `/mrcnn` represent the core of the Mask R-CNN code and have been
modified slightly to enable the repository from which they were taken to be used
with the DDSM data. `ddsm.py` is the file that represents the greatest departure from the repository above, as it was created to encapsulate the DDSM data as an object and closely align to COCO Dataset behavior. Much of the DDSM dataset functionality relies heavily on some of the COCO Python APIs.

## Requirements
- `pip`

To install the other requirements use
```bash
pip install -r requirements_mrcnn.txt
```

## How to run
Training and evaluation are run using `ddsm.py`
To use:
```bash
python ddsm.py <train | evaluate> --dataset <path_to_data> --model=<path_to_model | imagenet | coco>
```

## Notebooks
In addition to running the data script, the model can be used and inspected with the Jupyter notebooks in this directory. Specifically, `inspect_model.py` provides insight into the training/evaluation steps and the model results.
