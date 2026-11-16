"""
Confirmação da Fase 1: retreina os top-3 stems (ou 3 pontos da fronteira de Pareto) com
100 épocas, AINDA na validação (10% do treino do CIFAR-100), e grava o vencedor em
configs/cl_vit_stem_final.yaml.

Modos:
  --candidate i   treina só o candidato i (um índice do job array do Slurm)
  --finalize      lê os resultados dos candidatos e escolhe o stem final
  (sem flags)     faz tudo em sequência (uso local)
"""
import datetime
import json
import os

import optuna
import torchvision.datasets as datasets
import yaml

from utils import engine
from utils.dataloader import dataset_targets
from utils.pipeline import (DEFAULT_STEM_FINAL, common_parser, get_storage, load_yaml, recipe_config,
                            save_json, stem_cfg_from_params)
from utils.splits import get_or_create_split


def parse():
    p = common_parser(__doc__)
    p.add_argument('--study-name', default='phase1_stem_cifar100')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--candidate', type=int, default=None)
    p.add_argument('--finalize', action='store_true')
    p.add_argument('--output', default=DEFAULT_STEM_FINAL)
    return p.parse_args()


def select_candidates(study, k):
    complete = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f'{study.study_name}: nenhum trial COMPLETE.')
    if len(study.directions) > 1:
        # k pontos da fronteira de Pareto espalhados por FLOPs (menor custo ... maior acurácia)
        front = sorted(study.best_trials, key=lambda t: t.values[1])
        if len(front) <= k:
            chosen = front
        else:
            idx = sorted({round(i * (len(front) - 1) / (k - 1)) for i in range(k)})
            chosen = [front[i] for i in idx]
    else:
        chosen = sorted(complete, key=lambda t: t.value, reverse=True)[:k]
    return [{'trial': t.number, 'params': t.params, 'search_values': t.values} for t in chosen]


def main():
    args = parse()
    spaces = load_yaml(args.search_spaces)
    p1 = spaces['phase1']
    out_root = os.path.join(args.results_dir, 'confirm')
    study = optuna.load_study(study_name=args.study_name, storage=get_storage(args.storage_dir, args.study_name))
    cands = select_candidates(study, args.top_k)
    save_json(cands, os.path.join(out_root, 'candidates.json'))
    print(f'[confirm] candidatos: {[c["trial"] for c in cands]}')

    run_all = args.candidate is None and not args.finalize
    to_train = range(len(cands)) if run_all else ([args.candidate] if args.candidate is not None else [])
    if to_train:
        targets = dataset_targets(datasets.CIFAR100(root=args.data_path, train=True, download=True))
        split = get_or_create_split(p1['split_file'], targets, p1['val_frac'], p1['split_seed'])
    for i in to_train:
        if i >= len(cands):
            print(f'[confirm] candidato {i} não existe (só {len(cands)}); nada a fazer.')
            continue
        c = cands[i]
        cdir = os.path.join(out_root, f'cand_{i}_trial_{c["trial"]:04d}')
        if os.path.exists(os.path.join(cdir, 'result.json')):
            print(f'[confirm] {cdir} já concluído; pulando.')
            continue
        cfg = {**recipe_config(spaces, 'cl_vit', p1['dataset'], args),
               'stem_cfg': stem_cfg_from_params(c['params']), 'stem_lr_mult': c['params']['stem_lr_mult'],
               'epochs': args.epochs, 'eval_split': 'val', 'split_indices': split,
               'monitor': 'acc1', 'save_ckpt': 'best', 'out_dir': cdir}
        save_json({'candidate': c, 'config': {k: v for k, v in cfg.items() if k != 'split_indices'}},
                  os.path.join(cdir, 'config.json'))
        result = engine.train(cfg)
        save_json({'candidate': c, **{k: v for k, v in result.items() if k != 'history'}},
                  os.path.join(cdir, 'result.json'))

    if run_all or args.finalize:
        finalize(cands, out_root, args)


def finalize(cands, out_root, args):
    rows = []
    for i, c in enumerate(cands):
        path = os.path.join(out_root, f'cand_{i}_trial_{c["trial"]:04d}', 'result.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f'resultado do candidato {i} ausente: {path}')
        with open(path) as f:
            r = json.load(f)
        rows.append({'candidate': i, 'trial': c['trial'], 'params': c['params'],
                     'val_acc1_last_epoch': r['final']['acc1'], 'stem_params': r['stem_params'],
                     'stem_flops': r['stem_flops']})
    # Critério: acurácia de validação da ÚLTIMA época (100 épocas)
    win = max(rows, key=lambda r: r['val_acc1_last_epoch'])
    out = {
        'stem_cfg': stem_cfg_from_params(win['params']),
        'stem_lr_mult': float(win['params']['stem_lr_mult']),
        'source': {'study': args.study_name, 'trial': win['trial'],
                   'confirm_val_acc1_last_epoch': win['val_acc1_last_epoch'],
                   'confirm_epochs': args.epochs, 'stem_params': win['stem_params'], 'stem_flops': win['stem_flops'],
                   'created': datetime.datetime.now().isoformat(timespec='seconds')},
        'candidates': rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('# Gerado por confirm_top_stems.py: stem final do CL-ViT (não editar à mão)\n')
        yaml.safe_dump(out, f, sort_keys=False)
    save_json(rows, os.path.join(out_root, 'confirm_summary.json'))
    print(f'[confirm] stem final = trial {win["trial"]} (val acc {win["val_acc1_last_epoch"]:.2f}) -> {args.output}')


if __name__ == '__main__':
    main()
