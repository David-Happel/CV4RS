#!/bin/bash
#SBATCH -o /home/users/d/davidhappel/CV4RS/18t18b.out
#SBATCH --chdir=/home/users/d/davidhappel/CV4RS
#SBATCH -J CV4RS
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=gpu_short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla:1

echo "Loading venv..."
source /home/users/m/moran/venv/cv4rs/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "Executing..."
python3 train.py --epochs 50 --batch_size 10 --timepoints 6 --model trans --name 6t10b --no_process_data