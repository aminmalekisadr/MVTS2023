#!/bin/bash



#SBATCH --mail-user=mmalekis@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="MVVGE-auto_encoder"
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err




echo "Cuda device: $CUDA_VISIBLE_DEVICES"
echo "======= Start memory test ======="



python main.py   experiments/Config.yaml SMD auto_encoder