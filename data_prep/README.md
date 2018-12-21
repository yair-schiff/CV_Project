# Data Pre-processing

## Overview
This directory contains scripts that were used to process DDSM and INbreast data for use in classification and segmentation tasks.

## Requirements
- `pip`
- `ljpeg` directory: https://github.com/nicholaslocascio/ljpeg-ddsm (follow directions there to clone and compile necessary files)
To install the required packages use:
```bash
pip install -r requirements_data.txt
```

## How to use
### DDSM
The `data.py` script is used to process the DDSM data. As noted above, it depends on [this repository](https://github.com/nicholaslocascio/ljpeg-ddsm). This script will de-compress the `.LJPEG` images and read the `.OVERLAY` file to retrieve annotations.

### INbreast
The `inbreast.py` script will prepare the INbreast dataset. Note that this script assumes that in addition to the `.csv` provided by the data maintainers that an additional column is added to that `.csv` that indicates which images have masks.
