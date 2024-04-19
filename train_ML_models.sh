#!/bin/bash

#SBATCH --mem=50G
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu-long
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 22:00:00  # Job time limit
#SBATCH --array=0-9 # i.i.d, 10%, 50%, 90%
#SBATCH -o slurm-%j.out

source activate tf

## NOTE: Comment these out if you didn't install the cuda toolkit
# module load cuda/11
# module load cudnn/8

model_name="DeiT" # ViT, Swin, ConvNext, DeiT
no_classes=5 # 2 or 5
srun -n1 python roads_train_ML_models.py --model_name=$model_name --datasplit_parcent=$SLURM_ARRAY_TASK_ID --no_of_classes=$no_classes
