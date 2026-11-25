# Resumo do pipeline Optuna

## phase1_stem_cifar100

Trials: 25 ({'PRUNED': 17, 'COMPLETE': 8})

### Top-5

|   number |   values_val_acc1 | params_stem_channel_growth   |   params_stem_channels | params_stem_extra_conv   | params_stem_final_proj   |   params_stem_lr_mult |   user_attrs_final_acc1 |   user_attrs_final_auc_macro_ovr |   user_attrs_stem_params |   user_attrs_stem_flops |
|---------:|------------------:|:-----------------------------|-----------------------:|:-------------------------|:-------------------------|----------------------:|------------------------:|---------------------------------:|-------------------------:|------------------------:|
|       21 |             71.38 | double                       |                     48 | True                     | linear                   |                0.7887 |                   71.38 |                           0.9862 |                   144816 |                 9523200 |
|       24 |             71.32 | double                       |                     48 | True                     | linear                   |                0.7591 |                   71.32 |                           0.9866 |                   144816 |                 9523200 |
|       23 |             71.24 | double                       |                     48 | True                     | linear                   |                0.7596 |                   71.24 |                           0.9867 |                   144816 |                 9523200 |
|        0 |             70.78 | double                       |                     96 | False                    | none                     |                1      |                   70.78 |                           0.9868 |                   169056 |                11354112 |
|        4 |             70.18 | double                       |                     96 | False                    | none                     |                0.9184 |                   70.18 |                           0.9864 |                   169056 |                11354112 |

### Importância (fANOVA)

- stem_lr_mult: 0.979
- stem_extra_conv: 0.010
- stem_channels: 0.010
- stem_final_proj: 0.001
- stem_channel_growth: 0.000

## phase2_derma_cl_vit

Trials: 20 ({'PRUNED': 12, 'COMPLETE': 8})

### Top-5

|   number |   value |   values_0 | params_class_weighting   |   params_label_smoothing |   params_lr |   user_attrs_final_acc1 |   user_attrs_final_auc_macro_ovr |   user_attrs_final_recall_melanoma |   user_attrs_stem_params |   user_attrs_stem_flops |
|---------:|--------:|-----------:|:-------------------------|-------------------------:|------------:|------------------------:|---------------------------------:|-----------------------------------:|-------------------------:|------------------------:|
|       13 |  0.4992 |     0.4992 | sqrt_inverse_freq        |                 0.005245 |   0.0007312 |                   74.08 |                           0.9227 |                             0.4685 |                1.448e+05 |               9.523e+06 |
|        7 |  0.4962 |     0.4962 | sqrt_inverse_freq        |                 0.1364   |   0.00112   |                   73.88 |                           0.9251 |                             0.4775 |                1.448e+05 |               9.523e+06 |
|        2 |  0.495  |     0.495  | sqrt_inverse_freq        |                 0.01421  |   0.002214  |                   72.98 |                           0.9192 |                             0.4955 |                1.448e+05 |               9.523e+06 |
|        1 |  0.4829 |     0.4829 | sqrt_inverse_freq        |                 0.07669  |   0.001251  |                   73.38 |                           0.9239 |                             0.4595 |                1.448e+05 |               9.523e+06 |
|        9 |  0.4803 |     0.4803 | sqrt_inverse_freq        |                 0.07274  |   0.001378  |                   73.58 |                           0.9252 |                             0.4505 |                1.448e+05 |               9.523e+06 |

### Importância (fANOVA)

- label_smoothing: 0.530
- lr: 0.470
- class_weighting: 0.001

## phase2_derma_sl_vit

Trials: 20 ({'PRUNED': 12, 'COMPLETE': 8})

### Top-5

|   number |   value |   values_0 | params_class_weighting   |   params_label_smoothing |   params_lr |   user_attrs_final_acc1 |   user_attrs_final_auc_macro_ovr |   user_attrs_final_recall_melanoma |   user_attrs_stem_params |   user_attrs_stem_flops |
|---------:|--------:|-----------:|:-------------------------|-------------------------:|------------:|------------------------:|---------------------------------:|-----------------------------------:|-------------------------:|------------------------:|
|        1 |  0.5191 |     0.5191 | sqrt_inverse_freq        |                  0.07669 |   0.001251  |                   73.18 |                           0.9235 |                             0.5045 |                4.675e+04 |               3.026e+06 |
|        7 |  0.5094 |     0.5094 | sqrt_inverse_freq        |                  0.1364  |   0.00112   |                   73.58 |                           0.9256 |                             0.4595 |                4.675e+04 |               3.026e+06 |
|        9 |  0.4866 |     0.4866 | sqrt_inverse_freq        |                  0.07274 |   0.001378  |                   71.59 |                           0.9236 |                             0.5045 |                4.675e+04 |               3.026e+06 |
|        0 |  0.484  |     0.484  | none                     |                  0.08473 |   0.0008559 |                   75.97 |                           0.9257 |                             0.2883 |                4.675e+04 |               3.026e+06 |
|       19 |  0.4805 |     0.4805 | sqrt_inverse_freq        |                  0.1334  |   0.001788  |                   72.58 |                           0.9222 |                             0.5045 |                4.675e+04 |               3.026e+06 |

### Importância (fANOVA)

- label_smoothing: 0.623
- lr: 0.377
- class_weighting: 0.001

## phase2_derma_vit

Trials: 20 ({'PRUNED': 13, 'COMPLETE': 7})

### Top-5

|   number |   value |   values_0 | params_class_weighting   |   params_label_smoothing |   params_lr |   user_attrs_final_acc1 |   user_attrs_final_auc_macro_ovr |   user_attrs_final_recall_melanoma |   user_attrs_stem_params |   user_attrs_stem_flops |
|---------:|--------:|-----------:|:-------------------------|-------------------------:|------------:|------------------------:|---------------------------------:|-----------------------------------:|-------------------------:|------------------------:|
|        1 |  0.4574 |     0.4574 | sqrt_inverse_freq        |                  0.07669 |   0.001251  |                   71.68 |                           0.9112 |                             0.4595 |                     9408 |               5.898e+05 |
|        9 |  0.445  |     0.445  | sqrt_inverse_freq        |                  0.07274 |   0.001378  |                   72.38 |                           0.9197 |                             0.4144 |                     9408 |               5.898e+05 |
|        7 |  0.4333 |     0.4333 | sqrt_inverse_freq        |                  0.1364  |   0.00112   |                   70.19 |                           0.9141 |                             0.4505 |                     9408 |               5.898e+05 |
|        0 |  0.4259 |     0.4259 | none                     |                  0.08473 |   0.0008559 |                   74.08 |                           0.9219 |                             0.1441 |                     9408 |               5.898e+05 |
|        2 |  0.4159 |     0.4159 | sqrt_inverse_freq        |                  0.01421 |   0.002214  |                   70.99 |                           0.9062 |                             0.3333 |                     9408 |               5.898e+05 |

### Importância (fANOVA)

- label_smoothing: 0.547
- lr: 0.453
- class_weighting: 0.001

## Resultados finais (teste, última época, média ± desvio entre seeds)

### cifar100

| Modelo   |   Seeds | Acc top-1 (%)   | Acc@1 best (comparável c/ artigo)   |   Params (M) |   FLOPs (M) |   Vazão (img/s) |
|:---------|--------:|:----------------|:------------------------------------|-------------:|------------:|----------------:|
| cl_vit   |       3 | 74.94 ± 0.40    | 75.11 ± 0.26                        |         2.85 |       197.8 |            5435 |

### dermamnist

| Modelo   |   Seeds | Acc top-1 (%)   | Acc@1 best (comparável c/ artigo)   | F1 macro        | AUC macro (OvR)   | Acc balanceada   | Recall melanoma   |   Params (M) |   FLOPs (M) |   Vazão (img/s) |
|:---------|--------:|:----------------|:------------------------------------|:----------------|:------------------|:-----------------|:------------------|-------------:|------------:|----------------:|
| vit      |       3 | 74.41 ± 0.84    | 74.48 ± 1.10                        | 0.5490 ± 0.0087 | 0.9296 ± 0.0012   | 0.6094 ± 0.0155  | 0.4888 ± 0.0233   |         2.69 |       188.8 |            5527 |
| sl_vit   |       3 | 74.25 ± 0.10    | 74.33 ± 0.27                        | 0.5656 ± 0.0183 | 0.9295 ± 0.0030   | 0.6041 ± 0.0258  | 0.5336 ± 0.0162   |         2.73 |       191.3 |            4658 |
| cl_vit   |       3 | 75.41 ± 0.41    | 75.79 ± 0.76                        | 0.5610 ± 0.0054 | 0.9323 ± 0.0007   | 0.6140 ± 0.0080  | 0.4933 ± 0.0155   |         2.83 |       197.8 |            4675 |

### tiny_imagenet

| Modelo   |   Seeds | Acc top-1 (%)   | Acc@1 best (comparável c/ artigo)   |   Params (M) |   FLOPs (M) |   Vazão (img/s) |
|:---------|--------:|:----------------|:------------------------------------|-------------:|------------:|----------------:|
| cl_vit   |       3 | 61.87 ± 0.25    | 61.95 ± 0.13                        |          3.3 |       234.6 |            6979 |
