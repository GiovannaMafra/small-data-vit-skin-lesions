# ViTs com poucos dados em lesões de pele: reprodução de SPT + LSA e um híbrido convolucional (CL-ViT)

Projeto da disciplina **CIN0040 – Introdução à Aprendizagem Profunda** (CIn-UFPE, 2026.2).

> **Este repositório é um fork de [aanna0701/SPT_LSA_ViT](https://github.com/aanna0701/SPT_LSA_ViT)**, o código oficial do artigo *Vision Transformer for Small-Size Datasets* (Lee, Lee e Song, 2021). O código-base e as ideias de SPT e LSA pertencem aos autores originais. O README original está preservado em [`README_ORIGINAL.md`](README_ORIGINAL.md).

## Resumo

Como treinar Vision Transformers do zero com poucos dados, em um problema real de saúde? Reproduzimos os resultados do artigo (SPT + LSA) em CIFAR-100 e Tiny-ImageNet, propomos o **CL-ViT** (um *stem* convolucional no lugar do SPT, com LSA) e avaliamos os modelos também no **DermaMNIST** (lesões de pele, 7 classes, desbalanceado).

<!-- Quando houver resultados, coloque aqui uma figura-resumo (ex.: gráfico de barras ou mapas de atenção). -->

## Resultados

| Modelo | CIFAR-100 (acc. %) | Tiny-ImageNet (acc. %) | DermaMNIST (AUC / F1 macro) |
|---|---|---|---|
| ViT | _a preencher_ | _a preencher_ | _a preencher_ |
| SL-ViT (artigo) | _a preencher_ | _a preencher_ | _a preencher_ |
| **CL-ViT (nosso)** | _a preencher_ | _a preencher_ | _a preencher_ |

Valores do artigo, como referência: ViT 73,81 e SL-ViT 76,92 (CIFAR-100); ViT 57,07 e SL-ViT 61,07 (Tiny-ImageNet). Resultados nossos: média ± desvio de 3 sementes, acurácia da última época.

## O que é nosso e o que é original

| Parte | Origem |
|---|---|
| `main.py`, `models/` (ViT, SPT, LSA), `utils/` | Código original dos autores |
| `models/` (stem convolucional / CL-ViT) | Nossa contribuição |
| `utils/dataloader.py` (suporte ao DermaMNIST) | Nossa adaptação |
| `scripts/` (jobs Slurm), `notebooks/`, `results/` | Nossa contribuição |

## Estrutura

```
main.py, models/, utils/   código-base (autores) + nossas alterações
scripts/                   jobs Slurm (Cluster Apuana, CIn-UFPE)
notebooks/                 01_eda, 02_resultados, 03_demo_treino_curto (Colab)
results/                   um JSON/CSV por execução
```

## Como reproduzir

```bash
git clone <URL-DESTE-REPOSITORIO>
cd <pasta>
python -m venv venv && source venv/bin/activate
```

**1. PyTorch com CUDA.** `torch` e `torchvision` ficam fora do `requirements.txt` porque o wheel certo depende da versão de CUDA da máquina. Veja a versão máxima suportada pelo driver no canto superior direito de `nvidia-smi` e escolha o índice correspondente (lista completa em <https://pytorch.org/get-started/locally/>):

```bash
# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# só CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# confira se a GPU foi reconhecida
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**2. Demais dependências e execução.** Instale o torch primeiro: assim o `pip` não troca o wheel CUDA por outro ao resolver `timm`, `torchsummary` etc.

```bash
pip install -r requirements.txt
# exemplo (confira os argumentos com: python main.py --help)
python main.py --model vit --dataset CIFAR100 --lr 0.003
python main.py --model vit --dataset CIFAR100 --lr 0.001 --is_SPT --is_LSA
```

Os notebooks em `notebooks/` rodam no Google Colab e carregam os resultados de `results/`.

## Equipe

- Giovanna Marques Mafra
- Luma Rios Ferreira
- Mariana Sousa Freire

Professor: Tsang Ing Ren.

## Créditos e citação

Se usar este trabalho, cite também o artigo original:

```bibtex
@article{lee2021vision,
  title   = {Vision Transformer for Small-Size Datasets},
  author  = {Lee, Seung Hoon and Lee, Seunghyun and Song, Byung Cheol},
  journal = {arXiv preprint arXiv:2112.13492},
  year    = {2021}
}
```

Outras fontes: MedMNIST v2 (DermaMNIST), HAM10000, CIFAR-100 e Tiny-ImageNet (ver a proposta/relatório do projeto).