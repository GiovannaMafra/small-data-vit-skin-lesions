#!/bin/bash
#SBATCH --job-name=p1_stem
# cada índice = 1 worker no MESMO study (ajuste nº de workers)
#SBATCH --array=0-3
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
# Fase 1: busca do stem do CL-ViT no CIFAR-100. EXTRA_ARGS ex.: "--multiobjective"
source slurm/common.sh
python optuna_stem_search.py --data-path "$DATA_PATH" --storage-dir "$OPTUNA_DIR" --results-dir "$RESULTS_DIR" \
    --workers "$NUM_WORKERS" --worker-id "$SLURM_ARRAY_TASK_ID" \
    --n-trials "${N_TRIALS:-25}" --search-epochs "${SEARCH_EPOCHS:-30}" ${EXTRA_ARGS:-}
