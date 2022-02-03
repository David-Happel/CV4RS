#!/bin/bash
#SBATCH -o /home/users/m/moran/venv/CV4RS/trans_12t15b.out
#SBATCH --chdir=/home/users/m/moran/CV4RS
#SBATCH -J CV4RS
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=gpu_short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla:2

echo "Loading venv..."
# source /home/users/m/moran/venv/cv4rs/bin/activate
source /home/users/d/davidhappel/venv/cv4rs/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "Executing..."
python3 main.py --epochs 5 --batch_size 15 --timepoints 12 --model trans --name stephen_trans_12t15b --no_process_data