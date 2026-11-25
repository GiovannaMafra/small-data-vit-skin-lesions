# Resumo do pipeline Optuna

## Resultados finais (teste, última época, média ± desvio entre seeds)

### dermamnist

| Modelo   |   Seeds | Acc top-1 (%)   | Acc@1 best (comparável c/ artigo)   | F1 macro        | AUC macro (OvR)   | Acc balanceada   | Recall melanoma   |   Params (M) |   FLOPs (M) |   Vazão (img/s) |
|:---------|--------:|:----------------|:------------------------------------|:----------------|:------------------|:-----------------|:------------------|-------------:|------------:|----------------:|
| sl_vit   |       3 | 75.38 ± 0.37    | 75.83 ± 0.44                        | 0.5854 ± 0.0077 | 0.9186 ± 0.0044   | 0.6073 ± 0.0075  | 0.5202 ± 0.0078   |         2.73 |       191.3 |            5674 |
| cl_vit   |       3 | 75.93 ± 0.15    | 76.61 ± 0.22                        | 0.5807 ± 0.0163 | 0.9270 ± 0.0026   | 0.6180 ± 0.0260  | 0.5516 ± 0.0237   |         2.83 |       197.8 |            2286 |
