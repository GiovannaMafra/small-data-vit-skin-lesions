#!/bin/bash

#SBATCH --job-name=optuna_clvit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=optuna_%j.out
#SBATCH --error=optuna_%j.err

cd ~/small-data-vit-skin-lesions

source ../venv/bin/activate

python optuna_clvit.py