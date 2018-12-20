#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=classify_tumor_cv_project
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=yzs208@nyu.edu
#SBATCH --output=classify_tumor_cv_project.out.txt

# Load modules
module purge
module load cuda/9.0.176 cudnn/9.0v7.3.0.29 
# Activate the conda environment
source activate vision_project_classify 

#v2: No pre-trained weights and train all layers 
# Run classification training 
PYTHONPATH=$PYTHONPATH:. python /scratch/yzs208/CV_Project/classification_tumor.py  --data /scratch/jtb470/DDSM/data --model-results /scratch/yzs208/CV_Project/model_results_tumor_v2 --lr 0.001 --batch-size 2 --epochs 20 

# Close environemnt and purge modules
source deactivate
module purge
