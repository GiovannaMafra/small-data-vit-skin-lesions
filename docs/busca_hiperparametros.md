# Resultados do pipeline Optuna: Fases 1 e 2

Rodado no cluster Apuana (CIn/UFPE) em 21–22/11/2026, branch `pipeline-optuna`.
Métricas da **última época**, sempre na **validação**: o teste não foi usado nestas fases.
Os journals com todos os trials estão em `optuna_storage/`. O stem final está em `configs/cl_vit_stem_final.yaml`.

| Fase | O quê | Jobs | Duração |
|---|---|---|---|
| 1 | Busca do stem do CL-ViT, CIFAR-100, 25 trials × 30 épocas | 32582 | 4h52 (1 worker) |
| 1b | Confirmação dos top-3 stems, 100 épocas | 32583, 32584 | ~3h por candidato |
| 2 | Desbalanceamento no DermaMNIST, 20 trials × 30 épocas por modelo | 32585 (vit), 32586 (sl_vit), 32587 (cl_vit) | 42 min / 2h10 / 52 min |

Nenhum trial deu FAIL (OOM, NaN ou erro) em nenhuma fase.

---

## Fase 1: busca do stem do CL-ViT

- **Dados:** CIFAR-100, treino com 45.000 imagens e validação com 5.000 (10% estratificado do treino, `splits/cifar100_val10_seed0.json`).
- **Receita:** a do artigo, fixa (lr 3e-3). Só os parâmetros do stem são buscados.
- **Sampler e pruner:** TPE + MedianPruner (5 trials de startup, 10 épocas de warmup).
- **Restrição de custo:** um stem com params ou FLOPs > 4× o SPT do SL-ViT é descartado antes de treinar. A referência do SPT é 46.752 params e 3,03 M FLOPs.
- **Resultado:** 25 trials, sendo 8 COMPLETE, 11 PRUNED pelo MedianPruner e 6 PRUNED por custo.

### Trials completos (30 épocas)

| # | Canais | Crescimento | Conv extra | Projeção final | lr do stem (×) | Acc val | F1 | Params do stem | Custo vs SPT (params / FLOPs) |
|---|---|---|---|---|---|---|---|---|---|
| **21** | 48 | double | sim | linear | 0,789 | **71,38** | 0,711 | 144.816 | 3,10× / 3,15× |
| 24 | 48 | double | sim | linear | 0,759 | 71,32 | 0,710 | 144.816 | 3,10× / 3,15× |
| 23 | 48 | double | sim | linear | 0,760 | 71,24 | 0,710 | 144.816 | 3,10× / 3,15× |
| 0 (stem original da proposta) | 96 | double | não | none | 1,000 | 70,78 | 0,704 | 169.056 | 3,62× / 3,75× |
| 4 | 96 | double | não | none | 0,918 | 70,18 | 0,699 | 169.056 | 3,62× / 3,75× |
| 1 | 48 | constant | sim | linear | 1,099 | 68,92 | 0,687 | 52.464 | 1,12× / 1,19× |
| 3 | 48 | constant | não | conv1x1 | 1,850 | 67,44 | 0,671 | 31.632 | 0,68× / 0,75× |
| 2 | 32 | double | não | linear | 1,214 | 66,98 | 0,667 | 31.968 | 0,68× / 0,73× |

### Trials cortados pelo MedianPruner

| # | Stem (canais, crescimento, conv extra, projeção) | Cortado na época | Acc nesse ponto |
|---|---|---|---|
| 20 | 96, constant, não, conv1x1 | 15 | 56,46 |
| 11, 13, 17 | 96, double, não, none (stem original) | 12 | 52,66 / 52,44 / 52,12 |
| 6 | 96, constant, não, conv1x1 | 12 | 52,34 |
| 12 | 64, double, não, conv1x1 | 11 | 50,16 |
| 8, 19 | 64, constant, não, none | 11 | 50,44 / 50,42 |
| 10 | 48, double, não, none | 11 | 50,40 |
| 9 | 64, constant, não, linear | 11 | 48,92 |
| 5 | 32, double, não, linear | 11 | 48,74 |

**Descartados por custo (> 4× o SPT):** trials 7, 14, 15, 16, 18 e 22, com stems de 4,4× a 12× o custo do SPT.

### Confirmação (top-3, 100 épocas, mesma validação)

| Candidato | Trial | Acc val (última época) | Acc top-5 | F1 | AUC | Acc balanceada | Melhor época (acc) |
|---|---|---|---|---|---|---|---|
| **0 (vencedor)** | **21** | **74,16** | 91,40 | 0,740 | 0,986 | 0,742 | 98 (74,24) |
| 1 | 24 | 73,84 | 91,62 | 0,735 | 0,986 | 0,738 | 97 (73,94) |
| 2 | 23 | 73,60 | 91,76 | 0,734 | 0,986 | 0,736 | 96 (73,92) |

**Stem final** (`configs/cl_vit_stem_final.yaml`): 48 canais, `growth=double`, `extra_conv=true`, `final_proj=linear` e `stem_lr_mult≈0,789`. O CL-ViT final tem 2,85 M de params e 198 M de FLOPs.

### Cuidados na interpretação

- **A vantagem do stem escolhido está dentro do ruído.** Ele ganha do stem original por só +0,6 pp (71,38 contra 70,78). O próprio stem original, mudando apenas o `stem_lr_mult`, variou 0,6 pp (70,78 no #0 contra 70,18 no #4). O ganho concreto é o tamanho: **14% menos parâmetros no stem** (145 mil contra 169 mil) sem perder acurácia.
- **O pruner foi inconsistente com o stem original.** A mesma arquitetura terminou duas vezes (#0 e #4) e foi cortada três vezes (#11, #13 e #17).
- **Os 3 candidatos da confirmação são o mesmo desenho de stem.** Só muda o `stem_lr_mult` (0,76 a 0,79), então a diferença de 0,56 pp entre eles não compara arquiteturas. Se a Fase 1 for rodada de novo, vale fazer o `confirm_top_stems.py` descartar arquiteturas repetidas.

---

## Fase 2: desbalanceamento no DermaMNIST

- **Dados:** DermaMNIST oficial, com 7.007 imagens de treino e 1.003 de validação. As imagens são redimensionadas de 28 para 32 px.
- **Objetivo:** F1 macro na validação, na última época.
- **Espaço de busca:**
  - `lr`: de 1e-4 a 5e-3 (log);
  - `class_weighting`: `none`, `inverse_freq` ou `sqrt_inverse_freq`;
  - `label_smoothing`: de 0 a 0,2.
- **Comparabilidade:** mesmo espaço, mesmo nº de trials e mesma seed do sampler para os 3 modelos. Por isso vários trials têm os mesmos parâmetros entre modelos. O CL-ViT usa o stem final da Fase 1.
- **Resultado:** 20 trials por modelo. As tabelas abaixo mostram os trials completos. Os demais foram cortados entre as épocas 11 e 20, todos com F1 ≤ 0,45 no momento do corte.

### Melhor trial por modelo

| Modelo | Trial | F1 | AUC | Recall melanoma | Acc balanceada | Acc | lr | Ponderação | Label smoothing |
|---|---|---|---|---|---|---|---|---|---|
| vit | #1 | 0,457 | 0,911 | 0,459 | 0,517 | 71,7 | 1,25e-3 | sqrt_inverse_freq | 0,077 |
| **sl_vit** | #1 | **0,519** | 0,923 | **0,505** | **0,564** | 73,2 | 1,25e-3 | sqrt_inverse_freq | 0,077 |
| cl_vit | #13 | 0,499 | 0,923 | 0,468 | 0,538 | 74,1 | 7,31e-4 | sqrt_inverse_freq | 0,005 |

### vit (7 completos, 13 cortados)

| # | lr | Ponderação | Label smoothing | F1 | AUC | Recall melanoma | Acc balanceada | Acc |
|---|---|---|---|---|---|---|---|---|
| **1** | 1,25e-3 | sqrt_inverse_freq | 0,077 | **0,457** | 0,911 | 0,459 | 0,517 | 71,7 |
| 9 | 1,38e-3 | sqrt_inverse_freq | 0,073 | 0,445 | 0,920 | 0,414 | 0,494 | 72,4 |
| 7 | 1,12e-3 | sqrt_inverse_freq | 0,136 | 0,433 | 0,914 | 0,450 | 0,492 | 70,2 |
| 0 | 8,56e-4 | none | 0,085 | 0,426 | 0,922 | 0,144 | 0,400 | 74,1 |
| 2 | 2,21e-3 | sqrt_inverse_freq | 0,014 | 0,416 | 0,906 | 0,333 | 0,448 | 71,0 |
| 4 | 4,60e-3 | none | 0,024 | 0,368 | 0,914 | 0,081 | 0,360 | 73,0 |
| 3 | 1,41e-4 | inverse_freq | 0,174 | 0,361 | 0,866 | 0,486 | 0,470 | 56,4 |

### sl_vit (8 completos, 12 cortados)

| # | lr | Ponderação | Label smoothing | F1 | AUC | Recall melanoma | Acc balanceada | Acc |
|---|---|---|---|---|---|---|---|---|
| **1** | 1,25e-3 | sqrt_inverse_freq | 0,077 | **0,519** | 0,923 | 0,505 | 0,564 | 73,2 |
| 7 | 1,12e-3 | sqrt_inverse_freq | 0,136 | 0,509 | 0,926 | 0,459 | 0,548 | 73,6 |
| 9 | 1,38e-3 | sqrt_inverse_freq | 0,073 | 0,487 | 0,924 | 0,505 | 0,538 | 71,6 |
| 0 | 8,56e-4 | none | 0,085 | 0,484 | 0,926 | 0,288 | 0,444 | 76,0 |
| 19 | 1,79e-3 | sqrt_inverse_freq | 0,133 | 0,480 | 0,922 | 0,505 | 0,515 | 72,6 |
| 2 | 2,21e-3 | sqrt_inverse_freq | 0,014 | 0,467 | 0,921 | 0,505 | 0,515 | 70,8 |
| 4 | 4,60e-3 | none | 0,024 | 0,441 | 0,920 | 0,225 | 0,428 | 75,2 |
| 3 | 1,41e-4 | inverse_freq | 0,174 | 0,367 | 0,875 | 0,468 | 0,481 | 58,9 |

### cl_vit (8 completos, 12 cortados)

| # | lr | Ponderação | Label smoothing | F1 | AUC | Recall melanoma | Acc balanceada | Acc |
|---|---|---|---|---|---|---|---|---|
| **13** | 7,31e-4 | sqrt_inverse_freq | 0,005 | **0,499** | 0,923 | 0,468 | 0,538 | 74,1 |
| 7 | 1,12e-3 | sqrt_inverse_freq | 0,136 | 0,496 | 0,925 | 0,477 | 0,535 | 73,9 |
| 2 | 2,21e-3 | sqrt_inverse_freq | 0,014 | 0,495 | 0,919 | 0,495 | 0,540 | 73,0 |
| 1 | 1,25e-3 | sqrt_inverse_freq | 0,077 | 0,483 | 0,924 | 0,459 | 0,514 | 73,4 |
| 9 | 1,38e-3 | sqrt_inverse_freq | 0,073 | 0,480 | 0,925 | 0,450 | 0,517 | 73,6 |
| 4 | 4,60e-3 | none | 0,024 | 0,466 | 0,929 | 0,243 | 0,440 | 75,7 |
| 0 | 8,56e-4 | none | 0,085 | 0,433 | 0,927 | 0,243 | 0,395 | 74,4 |
| 3 | 1,41e-4 | inverse_freq | 0,174 | 0,391 | 0,883 | 0,595 | 0,528 | 57,8 |

### Recall por classe (melhor trial de cada modelo)

| Classe | vit | sl_vit | cl_vit |
|---|---|---|---|
| 0 ceratose actínica | 0,485 | 0,606 | 0,576 |
| 1 carcinoma basocelular | 0,692 | 0,654 | 0,712 |
| 2 ceratose benigna | 0,355 | 0,282 | 0,336 |
| 3 dermatofibroma | **0,000** | 0,250 | **0,083** |
| 4 melanoma | 0,459 | 0,505 | 0,468 |
| 5 nevo melanocítico | 0,844 | 0,863 | 0,875 |
| 6 lesão vascular | 0,786 | 0,786 | 0,714 |

Os nomes seguem a ordem padrão de classes do DermaMNIST (o código identifica o melanoma como classe 4). Confirmem no `medmnist` antes de usar no texto.

### O que a Fase 2 mostra

- **Ranking por F1:** sl_vit (0,519) > cl_vit (0,499) > vit (0,457). Nesta fase o **CL-ViT não superou o SL-ViT**. O resultado vem de 1 seed, na validação. A comparação que vale para o artigo é a da Fase 3: teste, 3 seeds.
- **`sqrt_inverse_freq` venceu nos três modelos.**
  - Sem ponderação (`none`), a acurácia é a mais alta (74–76%), mas o recall de melanoma cai para 0,08–0,29: o modelo passa a prever o nevo, a classe majoritária.
  - Com `inverse_freq` e lr baixo, o recall de melanoma chega a subir (até 0,595 no cl_vit), mas a acurácia cai para ~57%.
- **Dermatofibroma (classe 3) quase não é reconhecido** em nenhum modelo, com recall entre 0 e 0,25.

---

## Avisos para a Fase 3 e para o artigo

1. **As GPUs não foram iguais.** Os jobs caíram em RTX 3090 (cluster-node5) e A100 80GB (cluster-node7). O sl_vit rodou na A100 e ficou ~3× mais lento por época. Por isso o throughput medido (vit 12.889, sl_vit 5.862 e cl_vit 11.849 img/s) e os tempos de treino **não podem ser comparados entre modelos**. Se forem reportar eficiência, meçam os 3 modelos na mesma GPU, por exemplo com `--nodelist=cluster-node5`.
2. **A Fase 3 lê destes arquivos:**
   - `configs/cl_vit_stem_final.yaml`: o stem do cl_vit;
   - `optuna_storage/phase2_derma_<modelo>.log`: os melhores hiperparâmetros da Fase 2, via `--params-from study:phase2_derma_<modelo>`.

   Os dois já estão no repositório. Exemplo, para o cl_vit no DermaMNIST com 3 seeds:
   ```bash
   sbatch --array=0-2 --export=ALL,MODELS="cl_vit",DATASETS="dermamnist",SEEDS="0 1 2" slurm/final_train.sbatch
   ```
   O tamanho do array é `nº de modelos × nº de bases × nº de seeds − 1`.
3. **QoS `simple`:** o limite de GPUs simultâneas é do grupo inteiro da QoS (os jobs ficam pendentes com `QOSGrpGRES`), então os jobs de vocês disputam GPU entre si.
