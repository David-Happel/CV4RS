#!/bin/bash
#SBATCH -o /home/users/d/davidhappel/CV4RS/lstm_36t12b3l.out
#SBATCH --chdir=/home/users/d/davidhappel/CV4RS
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
python3 main.py --epochs 20 --batch_size 12 --timepoints 36 --model lstm --lstm_layers 3 --name lstm_36t12b3l --no_process_data