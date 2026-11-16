#!/bin/bash
#SBATCH --job-name=p1_confirm
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
