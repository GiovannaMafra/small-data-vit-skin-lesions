#!/bin/bash
#SBATCH --job-name=p3_final
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# TODO: confirmar sintaxe de GPU no Apuana
#SBATCH --gres=gpu:1
# TODO: partição (o script.sh atual usa: -p short-simple)
##SBATCH --partition=<PARTICAO>
# TODO: QoS (o script.sh atual usa: --qos=simple)
##SBATCH --qos=<QOS>
# TODO: tempo máximo permitido na partição
##SBATCH --time=<HH:MM:SS>
# Fase 3: array sobre o produto MODELS x DATASETS x SEEDS (listas separadas por espaço).
# Uso: sbatch --array=0-N --export=ALL,MODELS="vit sl_vit",DATASETS="cifar100",SEEDS="0 1 2" slurm/final_train.sbatch
source slurm/common.sh
read -ra M <<< "${MODELS:?}"; read -ra D <<< "${DATASETS:?}"; read -ra S <<< "${SEEDS:-0 1 2}"
i=$SLURM_ARRAY_TASK_ID
seed=${S[$(( i % ${#S[@]} ))]}; i=$(( i / ${#S[@]} ))
ds=${D[$(( i % ${#D[@]} ))]};   i=$(( i / ${#D[@]} ))
model=${M[$i]}
echo "[final] model=$model dataset=$ds seed=$seed"
python final_train.py --model "$model" --dataset "$ds" --seed "$seed" --data-path "$DATA_PATH" \
    --storage-dir "$OPTUNA_DIR" --results-dir "$RESULTS_DIR" --workers "$NUM_WORKERS" \
    --epochs "${FINAL_EPOCHS:-100}" ${EXTRA_ARGS:-}
