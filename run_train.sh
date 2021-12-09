#!/bin/bash
#SBATCH -o /home/users/d/davidhappel/log.out
#SBATCH -J test-distributed-training
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=50G
#SBATCH --partition=gpu

source /home/users/d/davidhappel/venv/cv4rs/bin/activate
module load nvidia/cuda/10.0

python -m train.py

