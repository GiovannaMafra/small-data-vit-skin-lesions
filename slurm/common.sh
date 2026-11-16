#!/bin/bash
# Ambiente comum a todos os jobs. Os jobs devem ser submetidos a partir da raiz do repositório
# (o run_pipeline.sh faz isso). Ajuste os TODOs para o Apuana.
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "=== job ${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} host=$(hostname) $(date) ==="
nvidia-smi || echo "[aviso] nvidia-smi indisponível"

# TODO: módulos do cluster, se necessário (ex.: module load ...). Não sei quais existem no Apuana.
# module load <...>

# TODO: ativação do ambiente. O script.sh atual usa um venv fora do repositório:
source "${VENV_ACTIVATE:-../venv/bin/activate}"
# (alternativa conda: source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>)

# TODO: caminhos no cluster
export DATA_PATH="${DATA_PATH:-./data}"                 # TODO: onde ficam cifar-100-python/, tiny-imagenet-200/, dermamnist.npz
export OPTUNA_DIR="${OPTUNA_DIR:-./optuna_storage}"     # TODO: diretório COMPARTILHADO (visível por todos os nós) com os journals
export RESULTS_DIR="${RESULTS_DIR:-./results}"          # TODO: saída (trials, checkpoints, resultados finais)
export NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
mkdir -p "$OPTUNA_DIR" "$RESULTS_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
