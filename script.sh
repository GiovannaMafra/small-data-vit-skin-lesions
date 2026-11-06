#!/bin/bash
#SBATCH --job-name=vit_T-IMNET     
#SBATCH --output=logs/%x_%j.out          
#SBATCH --error=logs/%x_%j.err          
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
#echo "Modelo: ViT | Dataset: CIFAR100"
echo "Modelo: ViT | Dataset: TinyImageNet"

#ViT Puro (Baseline)
#testar com seed diferentes (tirar a media depois)
#python main.py --model vit --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 42
#python main.py --model vit --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 43
#python main.py --model vit --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 44

#S-ViT (Apenas com SPT)
#python main.py --model vit --is_SPT --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 42
#python main.py --model vit --is_SPT --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 43
python main.py --model vit --is_SPT --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 44

#L-ViT (Apenas com LSA)
#python main.py --model vit --is_LSA --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 42
#python main.py --model vit --is_LSA --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 43
#python main.py --model vit --is_LSA --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 44

#SL-ViT (Modelo Completo do Artigo: SPT + LSA)
#python main.py --model vit --is_SPT --is_LSA --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 42
#python main.py --model vit --is_SPT --is_LSA --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 43
#python main.py --model vit --is_SPT --is_LSA --dataset T-IMNET --data_path ./data --epochs 100 --lr 0.003 --seed 44

echo "Treinamento finalizado em: $(date)"