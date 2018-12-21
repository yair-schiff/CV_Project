#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=v4_classify_tumor_cv_project
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=yzs208@nyu.edu
#SBATCH --output=v4_classify_tumor_cv_project.out.txt

# Load modules
module purge
module load cuda/9.0.176 cudnn/9.0v7.3.0.29 
# Activate the conda environment
source activate vision_project_classify 

#v1: Pre train; stage 1 just heads for 10 epoch; swithc to stage 2
#v2: No pre-trained weights and train all layers 
#v3: Use all images no pre-trained and train all layers
#v4: Use params specified in https://github.com/jimmyyhwu/ddsm-visual-primitives: SGD, pretrained, RGB, etc
# Run classification training 
PYTHONPATH=$PYTHONPATH:. python /scratch/yzs208/CV_Project/classification_tumor.py  --data /scratch/jtb470/DDSM/data --model-results /scratch/yzs208/CV_Project/model_results_tumor_v4 --lr 0.0001 --batch-size 16 --epochs 100  

# Close environemnt and purge modules
source deactivate
module purge
