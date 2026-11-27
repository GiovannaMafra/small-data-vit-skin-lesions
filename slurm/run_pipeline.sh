#!/bin/bash
# Submete o pipeline inteiro com dependências. Rode da RAIZ do repositório:
#   bash slurm/run_pipeline.sh
# Variáveis opcionais: SEEDS="0 1 2", P1_WORKERS=4, P2_WORKERS=2, EXTRA_P1="--multiobjective"
#
#   Estritamente sequencial (a QoS simple só permite ~2 GPUs simultâneas, então paralelizar fases
#   só faria os jobs disputarem a fila com a Fase 1, que é o caminho crítico):
#   Fase 1 (array) ─afterok─> confirmação (array 0-2) ─afterok─> finalize
#     ─afterok─> Fase 2 (vit, sl_vit, cl_vit) ─afterok(as três)─> Fase 3 (todos os modelos e bases)
#   Atenção: afterok => se uma task falhar, o que depende dela fica em DependencyNeverSatisfied
#   (corrija, cancele os pendentes e resubmeta; os scripts retomam do journal / pulam o que já terminou).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
SEEDS="${SEEDS:-0 1 2}"; NS=$(wc -w <<< "$SEEDS")
P1_WORKERS="${P1_WORKERS:-4}"; P2_WORKERS="${P2_WORKERS:-2}"

sub() { sbatch --parsable "$@"; }
final() {  # final <dependency|""> <MODELS> <DATASETS>
    local dep=$1 models=$2 datasets=$3
    local n=$(( $(wc -w <<< "$models") * $(wc -w <<< "$datasets") * NS - 1 ))
    sub ${dep:+--dependency=afterok:$dep} --array=0-$n \
        --export=ALL,MODELS="$models",DATASETS="$datasets",SEEDS="$SEEDS" slurm/final_train.sbatch
}

P1=$(sub --array=0-$((P1_WORKERS-1)) --export=ALL,EXTRA_ARGS="${EXTRA_P1:-}" slurm/phase1_worker.sbatch)
C=$(sub --dependency=afterok:$P1 --array=0-2 slurm/confirm_top_stems.sbatch)
F=$(sub --dependency=afterok:$C slurm/confirm_top_stems.sbatch)            # finalize -> configs/cl_vit_stem_final.yaml
echo "Fase 1: $P1 | confirmação: $C | finalize: $F"

P2_VIT=$(sub --dependency=afterok:$F --array=0-$((P2_WORKERS-1)) --export=ALL,MODEL=vit slurm/phase2_worker.sbatch)
P2_SL=$(sub --dependency=afterok:$F --array=0-$((P2_WORKERS-1)) --export=ALL,MODEL=sl_vit slurm/phase2_worker.sbatch)
P2_CL=$(sub --dependency=afterok:$F --array=0-$((P2_WORKERS-1)) --export=ALL,MODEL=cl_vit slurm/phase2_worker.sbatch)
echo "Fase 2: vit=$P2_VIT sl_vit=$P2_SL cl_vit=$P2_CL"
P2_ALL="$P2_VIT:$P2_SL:$P2_CL"

echo "Fase 3 vit/sl_vit CIFAR+Tiny: $(final "$P2_ALL" "vit sl_vit" "cifar100 tiny_imagenet")"
echo "Fase 3 cl_vit CIFAR+Tiny:     $(final "$P2_ALL" "cl_vit" "cifar100 tiny_imagenet")"
echo "Fase 3 Derma vit:             $(final "$P2_ALL" "vit" "dermamnist")"
echo "Fase 3 Derma sl_vit:          $(final "$P2_ALL" "sl_vit" "dermamnist")"
echo "Fase 3 Derma cl_vit:          $(final "$P2_ALL" "cl_vit" "dermamnist")"
echo "Acompanhe com: squeue -u \$USER"
