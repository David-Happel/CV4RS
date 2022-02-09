#!/bin/bash
#SBATCH -o /home/users/a/anastasia.1/CV4RS/a_lstm_6t15b.out
#SBATCH --chdir=/home/users/a/anastasia.1/CV4RS
#SBATCH -J CV4RS
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=gpu_short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla:2

echo "Loading venv..."
source /home/users/a/anastasia.1/venv/cv4rs/bin/activate
#source /home/users/d/davidhappel/venv/cv4rs/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "Executing..."
python3 main.py --epochs 5 --batch_size 15 --timepoints 6 --model lstm --lstm_layers 1 --name a_lstm_6t15b #--no_process_data