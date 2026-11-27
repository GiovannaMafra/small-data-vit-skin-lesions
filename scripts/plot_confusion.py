"""
Matriz de confusão normalizada por linha (classe verdadeira), média das seeds, DermaMNIST (teste).

Ex.: python scripts/plot_confusion.py --models sl_vit cl_vit --seeds 42 43 44
Lê results/final/dermamnist/<modelo>/seed<k>/final_metrics.json e grava figures/derma_cm_<modelos>.{png,pdf}.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

NAMES = {'vit': 'ViT', 's_vit': 'S-ViT', 'l_vit': 'L-ViT', 'sl_vit': 'SL-ViT', 'cl_vit': 'CL-ViT'}
# Ordem oficial do DermaMNIST (medmnist INFO['dermamnist']['label']), siglas do HAM10000
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# Rampa sequencial de um só matiz (azul), claro -> escuro
CMAP = LinearSegmentedColormap.from_list(
    'seq_blue', ['#f7fafe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
INK, INK_MUTED = '#1f1f1e', '#5f5e5a'


def mean_cm(results_dir, model, seeds):
    mats, counts = [], None
    for s in seeds:
        path = os.path.join(results_dir, 'final', 'dermamnist', model, f'seed{s}', 'final_metrics.json')
        cm = np.array(json.load(open(path))['test']['confusion_matrix'], dtype=float)
        counts = cm.sum(1)  # nº de exemplos por classe verdadeira (igual em todas as seeds)
        mats.append(cm / cm.sum(1, keepdims=True))
    m = np.stack(mats)
    return m.mean(0), m.std(0), counts.astype(int)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', default=['sl_vit', 'cl_vit'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--results-dir', default='results')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    out = args.out or os.path.join('figures', 'derma_cm_' + '_'.join(args.models))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_MUTED, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED})
    n = len(args.models)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.8, 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, args.models):
        mean, std, counts = mean_cm(args.results_dir, model, args.seeds)
        im = ax.imshow(mean, cmap=CMAP, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = mean[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if v > 0.5 else INK, fontweight='bold' if i == j else 'normal')
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), [f'{c} (n={k})' for c, k in zip(CLASSES, counts)])
        ax.set_xlabel('Classe predita')
        ax.set_ylabel('Classe verdadeira')
        ax.set_title(f'{NAMES.get(model, model)}  (média de {len(args.seeds)} seeds)', fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        print(f'\n{NAMES.get(model, model)}: média (desvio) por linha')
        print('        ' + ' '.join(f'{c:>12s}' for c in CLASSES))
        for i, c in enumerate(CLASSES):
            print(f'{c:>6s}  ' + ' '.join(f'{mean[i, j]:.3f}({std[i, j]:.3f})' for j in range(len(CLASSES))))
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label('Fração da classe verdadeira')
    cb.outline.set_visible(False)
    fig.suptitle('DermaMNIST (teste): matriz de confusão normalizada por linha', fontsize=11)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    print(f'\n-> {out}.png / {out}.pdf')


if __name__ == '__main__':
    main()
