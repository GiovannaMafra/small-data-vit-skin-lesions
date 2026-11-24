"""
Curvas de loss (treino e teste) do DermaMNIST, média ± desvio das seeds, lado a lado para dois
orçamentos de épocas (ex.: 100 e 200).

Ex.: python scripts/plot_loss_epochs.py --models sl_vit cl_vit --seeds 42 43 44 \
         --runs 100=results 200=results/derma_200ep
Lê <results>/final/dermamnist/<modelo>/seed<k>/history.csv e grava figures/derma_loss_<épocas>.{png,pdf}.

Atenção: a loss usa o critério de cada modelo (label smoothing e pesos de classe da Fase 2), então
o nível absoluto não é comparável entre modelos; compare a forma da curva e 100 x 200 no mesmo modelo.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'vit': 'ViT', 'sl_vit': 'SL-ViT', 'cl_vit': 'CL-ViT'}
# Paleta categórica (ordem fixa): slot 1 azul, slot 2 laranja, slot 3 aqua
COLORS = {'sl_vit': '#2a78d6', 'cl_vit': '#eb6834', 'vit': '#1baf7a'}
INK, INK_MUTED, GRID = '#1f1f1e', '#5f5e5a', '#e4e3df'


def load(results_dir, model, seeds, key):
    curves = []
    for s in seeds:
        path = os.path.join(results_dir, 'final', 'dermamnist', model, f'seed{s}', 'history.csv')
        with open(path) as f:
            curves.append([float(r[key]) for r in csv.DictReader(f)])
    a = np.array(curves)
    return np.arange(1, a.shape[1] + 1), a.mean(0), a.std(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', default=['sl_vit', 'cl_vit'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--runs', nargs='+', default=['100=results', '200=results/derma_200ep'],
                   help='rótulo=diretório de resultados, um por coluna')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    runs = [r.split('=', 1) for r in args.runs]
    out = args.out or os.path.join('figures', 'derma_loss_' + '_vs_'.join(lab for lab, _ in runs) + 'ep')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK_MUTED,
                         'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'axes.titlesize': 10})
    rows = [('train_loss', 'Loss de treino'), ('eval_loss', 'Loss de teste')]
    fig, axes = plt.subplots(len(rows), len(runs), figsize=(5.2 * len(runs), 3.3 * len(rows)),
                             sharey='row', constrained_layout=True)
    axes = np.array(axes).reshape(len(rows), len(runs))
    for c, (lab, rdir) in enumerate(runs):
        for r, (key, ylabel) in enumerate(rows):
            ax = axes[r, c]
            for m in args.models:
                x, mu, sd = load(rdir, m, args.seeds, key)
                ax.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, linewidth=0)
                ax.plot(x, mu, color=COLORS[m], linewidth=2, label=NAMES[m])
                if key == 'eval_loss':  # marca o mínimo da média
                    i = int(mu.argmin())
                    ax.plot(x[i], mu[i], 'o', ms=6, color=COLORS[m], mec='white', mew=1.5, zorder=5)
                    ax.annotate(f'mín. ép. {x[i]}', (x[i], mu[i]), xytext=(0, 9), textcoords='offset points',
                                ha='center', fontsize=8, color=INK_MUTED)
                print(f'{lab:>4s} ep | {NAMES[m]:6s} | {key:10s} | início {mu[0]:.3f} | mín {mu.min():.3f} '
                      f'(ép {int(mu.argmin()) + 1}) | final {mu[-1]:.3f} ± {sd[-1]:.3f}')
            ax.set_xlim(1, len(x))
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(f'{lab} épocas')
            if r == len(rows) - 1:
                ax.set_xlabel('Época')
            if c == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, loc='upper right')
    fig.suptitle(f'DermaMNIST: loss por época, média ± desvio de {len(args.seeds)} seeds', fontsize=11)
    fig.text(0.5, -0.02, 'A loss inclui o label smoothing e os pesos de classe de cada modelo '
             '(SL-ViT 0,077; CL-ViT 0,005): compare 100 × 200 épocas no mesmo modelo, '
             'não o nível absoluto entre modelos.', ha='center', fontsize=8, color=INK_MUTED)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {out}.png / {out}.pdf')


if __name__ == '__main__':
    main()
