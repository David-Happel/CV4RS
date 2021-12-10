#!/bin/bash
#SBATCH -o /home/users/d/davidhappel/CV4RS/log.out
#SBATCH -J test-distributed-training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=50G
#SBATCH --partition=gpu_short

source /home/users/d/davidhappel/venv/cv4rs/bin/activate
module load nvidia/cuda/10.0

python -m train.py

