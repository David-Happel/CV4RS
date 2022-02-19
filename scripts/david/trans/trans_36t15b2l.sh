#!/bin/bash
#SBATCH -o /home/users/d/davidhappel/CV4RS/trans_36t12b2l.out
#SBATCH --chdir=/home/users/d/davidhappel/CV4RS/
#SBATCH -J CV4RS
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=gpu_short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla:2

echo "Loading venv..."
source /home/users/d/davidhappel/venv/cv4rs/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "Executing..."
python3 main.py --epochs 50 --batch_size 12 --timepoints 36 --model trans --name trans_36t12b2l --trans_layers 2 --no_process_data