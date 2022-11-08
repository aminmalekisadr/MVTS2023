#!/bin/bash

#SBATCH --mail-user=mmalekis@uwaterloo.ca
##SBATCH --mail-user=nalamron@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="MVVGE-auto_encoder"
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --time=168:00:00
#SBATCH --mem=19000
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/stdout-%x_%j.log
#SBATCH --error=logs/stderr-%x_%j.log

echo "Cuda device: $CUDA_VISIBLE_DEVICES"
echo "======= Start memory test ======="

## Load modules
#module load gcc


python main.py   experiments/Config.yaml SMD auto_encoder
