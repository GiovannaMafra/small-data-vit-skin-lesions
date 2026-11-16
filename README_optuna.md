# Pipeline Optuna (Apuana / Slurm)

## Estratégia
1. A **receita do artigo é fixa** para todos os modelos e bases (AdamW, warmup, cosine, label smoothing, CutMix, Mixup, AutoAugment, stochastic depth, RA, random erasing; **lr 0.003** para ViT, SL-ViT e CL-ViT). Ela é definida em `utils/cli.py` e `configs/search_spaces.yaml`.
2. **Fase 1** (CIFAR-100, val = 10% estratificado do treino, índices em `splits/`): o Optuna busca **só o stem** do CL-ViT. O nº de convs de stride 2 é log2(patch): 2 em 32px e 3 em 64px. O trial é podado se o stem custar mais de `--max-stem-cost-ratio` (padrão 4.0) vezes o SPT. Depois, os top-3 são retreinados por 100 épocas na val e o vencedor vai para `configs/cl_vit_stem_final.yaml`.
3. **Fase 2** (DermaMNIST, val oficial): a mesma busca (`lr`, `class_weighting`, `label_smoothing`), com o mesmo nº de trials e a mesma seed, roda para `vit`, `sl_vit` e `cl_vit`. O objetivo é o F1 macro.
4. **Fase 3** (sem Optuna): 100 épocas × 3 seeds, treino completo e avaliação **uma vez no teste**, reportando a **última época**. Tiny-ImageNet não tem busca: o stem vem do CIFAR-100 e o "val" oficial é o teste.
5. Todas as fases usam o mesmo laço de treino (`utils/engine.py`, também usado pelo `main.py`).

## Passo a passo no cluster do CIn (Apuana)
Os valores entre `<...>` não dá para saber pelo repositório: confirme com a documentação do Apuana ou com quem administra o cluster.

**1. Entrar e clonar** (no nó de login)
```bash
ssh <usuario>@<endereco-de-login-do-apuana>
git clone <URL-DO-REPOSITORIO> small-data-vit-skin-lesions
cd small-data-vit-skin-lesions && git checkout <branch-do-pipeline>
```

**2. Ambiente** (uma vez só; segue o `comandos_venv.txt`, com o venv **fora** do repositório, em `../venv`)
```bash
cd .. && python3 -m venv venv && source venv/bin/activate && cd small-data-vit-skin-lesions
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # ajuste o CUDA ao driver dos nós
pip install einops timm medmnist "optuna>=4" fvcore pyyaml matplotlib tabulate colorama torchsummary tensorboard scikit-learn pandas
```

**3. Dados** (no nó de login, se os nós de computação não tiverem internet)
```bash
python download_data.py        # cria data/cifar-100-python, data/tiny-imagenet-200, data/dermamnist.npz
ls data/cifar-100-python       # precisa ter: train, test, meta
```

**4. Configurar os jobs**
- `slurm/common.sh`: `VENV_ACTIVATE` (padrão `../venv/bin/activate`), `DATA_PATH`, `OPTUNA_DIR` e `RESULTS_DIR`. O `OPTUNA_DIR` precisa estar num disco **visível por todos os nós** (ex.: o home ou o projeto); o `/tmp` do nó não serve.
- Cabeçalho de cada `slurm/*.sbatch`: descomente e preencha `--partition`, `--qos` e `--time`. O `script.sh` atual usa `-p short-simple --qos=simple`. Confira o tempo máximo dessa partição com `sinfo -s` / `scontrol show partition <nome>`: se um trial de 30 épocas não couber nesse tempo, use outra partição.
- Opcional: `#SBATCH -w <nó>` para fixar um nó, como no `script.sh`.

**5. Teste rápido antes de submeter tudo** (1 job curto, valida o ambiente e os caminhos)
```bash
mkdir -p logs
sbatch --array=0 --export=ALL,N_TRIALS=1,SEARCH_EPOCHS=1,EXTRA_ARGS="--study-name teste_cluster --max-train-samples 512 --max-eval-samples 256" slurm/phase1_worker.sbatch
tail -f logs/p1_stem_*_0.out      # deve mostrar nvidia-smi, "cuda True" e "Trial 0 finished"
rm -f "$OPTUNA_DIR"/teste_cluster.log*
```

**6. Submeter o pipeline**: `bash slurm/run_pipeline.sh` (detalhes na próxima seção).

**7. Acompanhar e baixar resultados**
```bash
squeue -u $USER                       # fila
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
python scripts/summarize.py --storage-dir $OPTUNA_DIR --results-dir $RESULTS_DIR --out reports   # pode rodar no login
# na sua máquina:
rsync -av <usuario>@<endereco-de-login-do-apuana>:~/small-data-vit-skin-lesions/reports ./
```
Opcional: `pip install optuna-dashboard`, depois `optuna-dashboard $OPTUNA_DIR/phase1_stem_cifar100.log --port 8080` no login e `ssh -L 8080:localhost:8080 <usuario>@<login>` na sua máquina.

## Antes de submeter
- Preencha os `# TODO` de `slurm/common.sh` (venv/conda, `DATA_PATH`, `OPTUNA_DIR` **compartilhado entre os nós**, `RESULTS_DIR`) e o cabeçalho dos `slurm/*.sbatch` (partição, QoS, `--time`, GPU).
- `pip install -r requirements.txt` no ambiente do cluster (precisa de `optuna>=4`, `fvcore`, `einops`, `timm`, `medmnist`).
- Os dados devem estar em `$DATA_PATH`: `cifar-100-python/` (completo, com train, test e meta), `tiny-imagenet-200/` (zip oficial, sem reorganizar) e `dermamnist.npz`.

## Ordem de submissão
Rode da raiz do repositório:
```bash
bash slurm/run_pipeline.sh          # submete tudo com --dependency=afterok
```
```
Fase 1 (array, 4 workers) → confirmação (array 0-2) → finalize (gera cl_vit_stem_final.yaml)
   → Fase 2 cl_vit → Fase 3 cl_vit Derma
   → Fase 3 cl_vit CIFAR-100 / Tiny
Imediatos: Fase 2 vit, sl_vit (→ Fase 3 Derma de cada um); Fase 3 vit, sl_vit em CIFAR-100 / Tiny
```
Variáveis: `SEEDS="0 1 2"`, `P1_WORKERS=4`, `P2_WORKERS=2`, `EXTRA_P1="--multiobjective"`.
Jobs avulsos:
```bash
sbatch --array=0-3 slurm/phase1_worker.sbatch
sbatch --array=0-2 slurm/confirm_top_stems.sbatch && sbatch slurm/confirm_top_stems.sbatch   # candidatos, depois finalize
sbatch --array=0-1 --export=ALL,MODEL=sl_vit slurm/phase2_worker.sbatch
sbatch --array=0-5 --export=ALL,MODELS="vit sl_vit",DATASETS="cifar100",SEEDS="0 1 2" slurm/final_train.sbatch
```

## Acompanhar e retomar
- `squeue -u $USER`, `tail -f logs/<job>_<A>_<a>.out`
- Os studies ficam em `$OPTUNA_DIR/<study>.log` (JournalStorage): `phase1_stem_cifar100` e `phase2_derma_{vit,sl_vit,cl_vit}`.
  ```bash
  optuna trials --storage-class JournalFileBackend --storage $OPTUNA_DIR/phase2_derma_vit.log --study-name phase2_derma_vit   # ou:
  python scripts/summarize.py --storage-dir $OPTUNA_DIR --results-dir $RESULTS_DIR --out reports
  ```
- **Retomar:** é só ressubmeter o mesmo sbatch (`load_if_exists=True`). `--n-trials` é o total do study: os workers param quando há `n` trials COMPLETE+PRUNED. Trials de um worker morto ficam RUNNING e não entram nessa conta, então o orçamento é reposto. A Fase 3 pula runs que já têm `final_metrics.json`.
- Se um worker da Fase 1 morrer (ex.: estouro de tempo), o `afterok` não dispara. Nesse caso, ressubmeta o worker e, quando o study fechar, submeta a confirmação e o finalize à mão (comandos acima).
- OOM vira trial FAIL e NaN vira PRUNED; o worker continua nos dois casos (motivo em `user_attrs.fail_reason`).
- Smoke test local (CPU): `python optuna_stem_search.py --n-trials 2 --search-epochs 1 --max-train-samples 256 --max-eval-samples 128 --workers 0 --storage-dir /tmp/o --results-dir /tmp/r`

## Onde ficam os resultados (`$RESULTS_DIR`)
| Caminho | Conteúdo |
|---|---|
| `phase1/trial_XXXX/` | `config.json`, `history.csv/json` (por época), `best.pth`, `result.json` (custo, métricas da última época) |
| `phase1/pareto_front.json` | fronteira de Pareto (`--multiobjective`) |
| `confirm/` | candidatos, 100 épocas de cada um, `confirm_summary.json` → `configs/cl_vit_stem_final.yaml` |
| `phase2/<modelo>/trial_XXXX/` | igual à Fase 1. No study: F1, AUC, acc, recall por classe/melanoma e matriz de confusão (`user_attrs`) |
| `final/<base>/<modelo>/seed<k>/` | `final_metrics.json` (teste, última época, params, FLOPs, img/s, métricas desbalanceadas e matriz de confusão), `history.csv/json`, `last.pth` |
| `reports/` (summarize) | `summary.md`, top-k, importância, `history/importance/pareto.png`, `final_table_<base>.csv` (média ± desvio) |
