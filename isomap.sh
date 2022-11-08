#!/bin/bash



#SBATCH --mail-user=mmalekis@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="MVVGE-isomap"
#SBATCH --partition=gpu_pr1_long
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=168:00:00
#SBATCH --mem=10000
#SBATCH --gres=gpu:p100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err




echo "Cuda device: $CUDA_VISIBLE_DEVICES"
echo "======= Start memory test ======="



python main.py experiments/Config.yaml SMD tsne
