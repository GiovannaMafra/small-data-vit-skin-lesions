#!/bin/bash
#SBATCH --job-name=sl_vit_cifar100_seed44
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p short-simple
#SBATCH --qos=simple
#SBATCH --signal=USR1@5
#SBATCH --nodelist=cluster-node5

# Ir para a pasta onde o job foi enviado
cd "$SLURM_SUBMIT_DIR"

# Criar pasta de logs, caso ainda não exista
mkdir -p logs

# Ativar o ambiente virtual
export VIRTUAL_ENV=/home/CIN/lrf2/venv/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Informações do treinamento
echo "========================================"
echo "Iniciando o treinamento em: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Modelo: SL-ViT | Dataset: CIFAR-100 | Seed: 44"
echo "========================================"

# Verificar a GPU disponibilizada pelo SLURM
nvidia-smi

# Verificar PyTorch e CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma GPU disponível')"

# Verificar Colorama
python -c "from colorama import Fore, Style; print('Colorama: OK')"

# SL-ViT: SPT + LSA
python main.py \
    --model vit \
    --is_SPT \
    --is_LSA \
    --dataset CIFAR100 \
    --data_path ./data \
    --epochs 100 \
    --lr 0.001 \
    --seed 44

# Capturar código de saída do treinamento
STATUS=$?

echo "========================================"
echo "Treinamento encerrado em: $(date)"
echo "Exit code: $STATUS"
echo "========================================"

exit $STATUS
