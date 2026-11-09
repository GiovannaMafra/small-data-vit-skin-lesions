#!/bin/bash

#SBATCH --job-name=slvit_seed43
#SBATCH --output=logs/slvit_seed43_%j.out
#SBATCH --error=logs/slvit_seed43_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --signal=USR1@5
#SBATCH --exclude=cluster-node9

cd /home/CIN/lrf2/small-data-vit-skin-lesions
mkdir -p logs

export VIRTUAL_ENV=/home/CIN/lrf2/venv/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r

echo "=========================================="
echo "Experimento: SL-ViT (SPT + LSA)"
echo "Dataset: CIFAR100"
echo "Seed: 43"
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

echo "Iniciando treinamento SL-ViT..."

python main.py \
    --model vit \
    --dataset CIFAR100 \
    --lr 0.003 \
    --is_SPT \
    --is_LSA \
    --data_path ./data \
    --epochs 100 \
    --seed 43

STATUS=$?

echo "=========================================="
echo "Treinamento finalizado."
echo "Exit code: $STATUS"
echo "=========================================="

exit $STATUS
EOF
