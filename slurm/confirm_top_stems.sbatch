#!/bin/bash
#SBATCH --job-name=p1_confirm
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
# Confirmação dos top-3 stems (100 épocas na validação).
#  - como job array (--array=0-2): cada índice treina um candidato
#  - sem array: --finalize (escolhe o vencedor e grava configs/cl_vit_stem_final.yaml)
source slurm/common.sh
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODE="--candidate $SLURM_ARRAY_TASK_ID"
else
    MODE="--finalize"
fi
python confirm_top_stems.py $MODE --data-path "$DATA_PATH" --storage-dir "$OPTUNA_DIR" \
    --results-dir "$RESULTS_DIR" --workers "$NUM_WORKERS" --epochs "${CONFIRM_EPOCHS:-100}" ${EXTRA_ARGS:-}
