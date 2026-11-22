#!/bin/bash
#SBATCH --job-name=p1_stem
# cada índice = 1 worker no MESMO study (ajuste nº de workers)
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# GPU (Apuana: todas as partições têm GPU; sintaxe gres/gpu)
#SBATCH --gres=gpu:1
# Partição: short-simple (MaxTime 2-00:00:00, exige QoS simple). A padrão (debug) só tem 30 min.
#SBATCH --partition=short-simple
#SBATCH --qos=simple
# Tempo: limite da short-simple (os scripts retomam do journal do Optuna / pulam o que já terminou)
#SBATCH --time=2-00:00:00
# Fase 1: busca do stem do CL-ViT no CIFAR-100. EXTRA_ARGS ex.: "--multiobjective"
source slurm/common.sh
python optuna_stem_search.py --data-path "$DATA_PATH" --storage-dir "$OPTUNA_DIR" --results-dir "$RESULTS_DIR" \
    --workers "$NUM_WORKERS" --worker-id "$SLURM_ARRAY_TASK_ID" \
    --n-trials "${N_TRIALS:-25}" --search-epochs "${SEARCH_EPOCHS:-30}" ${EXTRA_ARGS:-}
