#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=normals_cv_project_data_prep
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=yzs208@nyu.edu
#SBATCH --output=normals_cv_project_data_prep.out.txt

# Load modules
module load anaconda3/4.3.1 
# Activate the conda environment
source activate vision_project 

# Run the evaluation script
PYTHONPATH=$PYTHONPATH:. python data.py --cases /scratch/jtb470/DDSM/cases_normals --data /scratch/jtb470/DDSM/data_normals 

# Close environemnt and purge modules
source deactivate
module purge
