#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=classifycv_project
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=yzs208@nyu.edu
#SBATCH --output=classify_cv_project.out.txt

# Load modules
module purge
module load cuda/9.0.176 cudnn/9.0v7.3.0.29 
# Activate the conda environment
source activate vision_project_classify 

# Run classification training 
PYTHONPATH=$PYTHONPATH:. python /scratch/yzs208/CV_Project/classification.py  --data /scratch/jtb470/DDSM/data --model-results model_results --batch-size 2 --epochs 20 

# Close environemnt and purge modules
source deactivate
module purge