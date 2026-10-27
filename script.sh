#!/bin/bash
#SBATCH --job-name=l_svit_cifar100       #mudar aqui o nome do modelo que está treinando
#SBATCH --output=logs/%x_%j.out          #arquivo onde serão salvas as métricas (stdout)
#SBATCH --error=logs/%x_%j.err           #arquivo onde serão salvos eventuais erros (stderr)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p short-simple
#SBATCH --qos=simple
#SBATCH --signal=USR1@5

# 1. Garante que está no diretório correto do projeto
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# 2. Ativa o ambiente virtual do Python
source ../venv/bin/activate

# 3. Executa o treinamento do modelo
# (Altere as flags abaixo conforme o modelo que deseja treinar)

echo "Iniciando o treinamento em: $(date)"
echo "Modelo: ViT | Dataset: CIFAR-100"

#ViT Puro (Baseline)
#python main.py --model vit --dataset CIFAR100 --data_path ./data --epochs 100 --lr 0.003 --seed 42

#S-ViT (Apenas com SPT)
#python main.py --model vit --is_SPT --dataset CIFAR100 --data_path ./data --epochs 100 --lr 0.001 --seed 42

#L-ViT (Apenas com LSA)
python main.py --model vit --is_LSA --dataset CIFAR100 --data_path ./data --epochs 100 --lr 0.001 --seed 42

#SL-ViT (Modelo Completo do Artigo: SPT + LSA)
#python main.py --model vit --is_SPT --is_LSA --dataset CIFAR100 --data_path ./data --epochs 100 --lr 0.001 --seed 42

echo "Treinamento finalizado em: $(date)"