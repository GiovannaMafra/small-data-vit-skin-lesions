#!/bin/bash

#SBATCH --job-name=lvit_seed44
#SBATCH --output=logs/lvit_seed44_%j.out
#SBATCH --error=logs/lvit_seed44_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --signal=USR1@5
#SBATCH --exclude=cluster-node9

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export VIRTUAL_ENV=/home/CIN/lrf2/venv/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r

echo "=========================================="
echo "Experimento: L-ViT (LSA)"
echo "Dataset: CIFAR100"
echo "Seed: 44"
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

echo "Iniciando treinamento L-ViT..."

python main.py \
    --model vit \
    --is_LSA \
    --dataset CIFAR100 \
    --data_path ./data \
    --epochs 100 \
    --lr 0.003 \
    --seed 44

STATUS=$?

echo "=========================================="
echo "Treinamento finalizado."
echo "Exit code: $STATUS"
echo "=========================================="

exit $STATUS
