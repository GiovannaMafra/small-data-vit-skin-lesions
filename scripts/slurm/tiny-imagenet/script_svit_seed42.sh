#!/bin/bash

#SBATCH --job-name=svit_seed42
#SBATCH --output=logs/tiny-imagenet/svit_seed42_%j.out
#SBATCH --error=logs/tiny-imagenet/svit_seed42_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --signal=USR1@5
#SBATCH --exclude=cluster-node9

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/tiny-imagenet

export VIRTUAL_ENV=/home/CIN/lrf2/venv/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r

echo "=========================================="
echo "Experimento: S-ViT (SPT)"
echo "Dataset: Tiny ImageNet"
echo "Seed: 42"
echo "Node: $(hostname)"
echo "=========================================="

python --version
which python

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA disponível:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma GPU disponível')
"

python -c "
import torch
import torchvision
import torchsummary
import numpy
import einops
import timm
import tensorboard
import tqdm
import colorama
print('Todas as dependências OK')
"

echo "Iniciando treinamento S-ViT..."

python main.py \
    --model vit \
    --is_SPT \
    --dataset T-IMNET \
    --data_path ./data \
    --epochs 100 \
    --lr 0.001 \
    --seed 42

STATUS=$?

echo "=========================================="
echo "Treinamento finalizado."
echo "Exit code: $STATUS"
echo "=========================================="

exit $STATUS