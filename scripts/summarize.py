"""
Resumo do pipeline: top-k trials, importância dos hiperparâmetros, gráficos (PNG, sem display)
e tabela consolidada dos resultados finais (média ± desvio entre seeds).

Ex.: python scripts/summarize.py --storage-dir $OPTUNA_DIR --results-dir results --out reports
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import get_storage  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def study_names(storage_dir):
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(storage_dir, '*.log')))


def _save(ax, path):
    fig = np.ravel(ax)[0].figure  # os plots do optuna devolvem Axes (ou array de Axes)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close('all')


def summarize_study(name, storage_dir, out_dir, top_k):
    study = optuna.load_study(study_name=name, storage=get_storage(storage_dir, name))
    sdir = os.path.join(out_dir, name)
    os.makedirs(sdir, exist_ok=True)
    df = study.trials_dataframe(attrs=('number', 'value', 'values', 'params', 'state', 'user_attrs', 'duration'))
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(os.path.join(sdir, 'trials.csv'), index=False)
    multi = len(study.directions) > 1
    states = df['state'].value_counts().to_dict()
    lines = [f'## {name}', '', f'Trials: {len(df)} ({states})', '']

    complete = df[df['state'] == 'COMPLETE']
    if len(complete):
        # nomes variam: value | values_0, values_1 | values_<metric_name> (set_metric_names)
        value_cols = [c for c in df.columns if c == 'value' or c.startswith('values')]
        sort_col = value_cols[0]
        top = complete.sort_values(sort_col, ascending=False).head(top_k)
        cols = ['number'] + value_cols + \
               [c for c in df.columns if c.startswith('params_')] + \
               [c for c in ('user_attrs_final_acc1', 'user_attrs_final_auc_macro_ovr', 'user_attrs_final_recall_melanoma',
                            'user_attrs_stem_params', 'user_attrs_stem_flops') if c in df.columns]
        lines += [f'### Top-{top_k}', '', top[cols].to_markdown(index=False, floatfmt='.4g'), '']

    target = (lambda t: t.values[0]) if multi else None
    tname = 'objective 0' if multi else 'objective'
    plots = {
        'history.png': lambda: optuna.visualization.matplotlib.plot_optimization_history(
            study, target=target, target_name=tname),
        'importance.png': lambda: optuna.visualization.matplotlib.plot_param_importances(
            study, target=target, target_name=tname),
    }
    if multi:
        plots['pareto.png'] = lambda: optuna.visualization.matplotlib.plot_pareto_front(study)
    for fname, fn in plots.items():
        try:
            _save(fn(), os.path.join(sdir, fname))
        except Exception as e:  # ex.: poucos trials para importância
            print(f'[summarize] {name}/{fname}: {e}')
    try:
        imp = optuna.importance.get_param_importances(study, target=target)
        lines += ['### Importância (fANOVA)', '', *[f'- {k}: {v:.3f}' for k, v in imp.items()], '']
        with open(os.path.join(sdir, 'importance.json'), 'w') as f:
            json.dump(imp, f, indent=1)
    except Exception as e:
        lines += [f'_importância indisponível: {e}_', '']
    if multi:
        front = [{'number': t.number, 'values': t.values, 'params': t.params} for t in study.best_trials]
        with open(os.path.join(sdir, 'pareto_front.json'), 'w') as f:
            json.dump(front, f, indent=1)
    return '\n'.join(lines)


FINAL_METRICS = [('acc1', 'Acc top-1 (%)'), ('f1_macro', 'F1 macro'), ('auc_macro_ovr', 'AUC macro (OvR)'),
                 ('balanced_acc', 'Acc balanceada'), ('recall_melanoma', 'Recall melanoma')]


def final_table(results_dir, out_dir):
    import pandas as pd
    rows = []
    for path in glob.glob(os.path.join(results_dir, 'final', '*', '*', 'seed*', 'final_metrics.json')):
        with open(path) as f:
            r = json.load(f)
        row = {'dataset': r['dataset'], 'model': r['model'], 'seed': r['seed'],
               'params': r['model_params'], 'flops': r['model_flops'], 'throughput': r['throughput_img_s']}
        for k, _ in FINAL_METRICS:
            if k in r['test']:
                row[k] = r['test'][k]
        best = r.get('test_best') or {}
        if 'eval_acc1' in best:  # época escolhida pelo monitor (acc1; f1_macro no Derma)
            row['acc1_best'] = best['eval_acc1']
        rows.append(row)
    if not rows:
        return '_sem resultados finais ainda_'
    df = pd.DataFrame(rows).sort_values(['dataset', 'model', 'seed'])
    df.to_csv(os.path.join(out_dir, 'final_runs.csv'), index=False)

    order = {'vit': 0, 's_vit': 1, 'l_vit': 2, 'sl_vit': 3, 'cl_vit': 4}
    out = []
    for ds, g in df.groupby('dataset'):
        metrics = ['acc1'] if ds != 'dermamnist' else [k for k, _ in FINAL_METRICS]
        table = []
        for model, gm in sorted(g.groupby('model'), key=lambda x: order.get(x[0], 9)):
            r = {'Modelo': model, 'Seeds': len(gm)}
            for k in metrics:
                if k in gm:
                    v = gm[k].astype(float)
                    r[dict(FINAL_METRICS)[k]] = f'{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}' \
                        if k == 'acc1' else f'{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}'
                if k == 'acc1' and 'acc1_best' in gm and gm['acc1_best'].notna().any():
                    v = gm['acc1_best'].dropna().astype(float)
                    r['Acc@1 best (comparável c/ artigo)'] = \
                        f'{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}'
            r['Params (M)'] = f"{gm['params'].iloc[0] / 1e6:.2f}"
            r['FLOPs (M)'] = f"{gm['flops'].iloc[0] / 1e6:.1f}"
            r['Vazão (img/s)'] = f"{gm['throughput'].mean():.0f}"
            table.append(r)
        t = pd.DataFrame(table)
        t.to_csv(os.path.join(out_dir, f'final_table_{ds}.csv'), index=False)
        out += [f'### {ds}', '', t.to_markdown(index=False), '']
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage-dir', default=os.environ.get('OPTUNA_DIR', './optuna_storage'))
    p.add_argument('--results-dir', default=os.environ.get('RESULTS_DIR', './results'))
    p.add_argument('--out', default='reports')
    p.add_argument('--studies', nargs='*', default=None, help='padrão: todos os journals em --storage-dir')
    p.add_argument('--top-k', type=int, default=5)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = ['# Resumo do pipeline Optuna', '']
    for name in (a.studies or study_names(a.storage_dir)):
        try:
            md.append(summarize_study(name, a.storage_dir, a.out, a.top_k))
        except Exception as e:
            md.append(f'## {name}\n\n_erro: {e}_\n')
    md += ['## Resultados finais (teste, última época, média ± desvio entre seeds)', '',
           final_table(a.results_dir, a.out)]
    with open(os.path.join(a.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(md))
    print(f'[summarize] -> {a.out}/summary.md')


if __name__ == '__main__':
    main()
