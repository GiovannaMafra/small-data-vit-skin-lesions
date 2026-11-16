"""
Fase 1: busca do desenho do stem do CL-ViT no CIFAR-100 (val = 10% estratificado do treino).

Só os parâmetros do stem são buscados; a receita do artigo fica fixa.
Vários workers (job array) podem rodar este script em paralelo sobre o mesmo study.

Ex.: python optuna_stem_search.py --n-trials 25 --search-epochs 30
     python optuna_stem_search.py --multiobjective          # (acc ↑, FLOPs do stem ↓), NSGA-II
"""
import os

import optuna
import torchvision.datasets as datasets

from models.create_model import create_model
from utils import engine
from utils.cost import spt_reference_cost, stem_cost
from utils.dataloader import dataset_targets
from utils.pipeline import (add_optuna_args, common_parser, load_yaml, make_study, optimize, recipe_config,
                            run_guarded, save_json, stem_cfg_from_params, suggest, trial_metrics_attrs)
from utils.splits import get_or_create_split

STUDY = 'phase1_stem_cifar100'
IMG_SIZE, N_CLASSES = 32, 100


def parse():
    p = common_parser(__doc__)
    add_optuna_args(p, n_trials=25, epochs=30)
    p.add_argument('--study-name', default=STUDY)
    p.add_argument('--max-stem-cost-ratio', type=float, default=4.0,
                   help='prune se params OU FLOPs do stem > ratio x custo do SPT do SL-ViT')
    p.add_argument('--multiobjective', action='store_true')
    return p.parse_args()


def check_stem(stem_cfg, max_ratio, base_cfg):
    """Constrói o CL-ViT, compara tokens/custo do stem com o SPT do SL-ViT."""
    args = engine.make_args({**base_cfg, 'model': 'cl_vit', 'stem_cfg': stem_cfg})
    model = create_model(IMG_SIZE, N_CLASSES, args)
    patch = 4 if IMG_SIZE == 32 else 8  # mesma regra de models/create_model.py
    c = stem_cost(model.to_patch_embedding, IMG_SIZE)
    ref = spt_reference_cost(IMG_SIZE, patch, dim=model.dim)
    assert c['n_tokens'] == ref['n_tokens'] == model.num_patches, \
        f"nº de tokens do stem ({c['n_tokens']}) != SL-ViT ({ref['n_tokens']})"
    info = {'stem_params': c['params'], 'stem_flops': c['flops'], 'n_tokens': c['n_tokens'],
            'num_stride2_convs': model.to_patch_embedding.num_convs,
            'spt_params': ref['params'], 'spt_flops': ref['flops'],
            'stem_params_ratio': c['params'] / ref['params'], 'stem_flops_ratio': c['flops'] / ref['flops']}
    info['over_budget'] = max(info['stem_params_ratio'], info['stem_flops_ratio']) > max_ratio
    return info


def main():
    args = parse()
    spaces = load_yaml(args.search_spaces)
    p1 = spaces['phase1']
    out_root = os.path.join(args.results_dir, 'phase1')
    directions = ['maximize', 'minimize'] if args.multiobjective else ['maximize']
    study, wid = make_study(args, args.study_name, directions, args.multiobjective)
    study.set_metric_names(['val_acc1', 'stem_flops'] if args.multiobjective else ['val_acc1'])

    # Split estratificado fixo (índices salvos em arquivo); o teste não é tocado.
    targets = dataset_targets(datasets.CIFAR100(root=args.data_path, train=True, download=True))
    split = get_or_create_split(p1['split_file'], targets, p1['val_frac'], p1['split_seed'])

    if wid == 0:  # só um worker enfileira o stem padrão da proposta
        study.enqueue_trial(p1['default_trial'], skip_if_exists=True)

    base_cfg = recipe_config(spaces, 'cl_vit', p1['dataset'], args)

    def objective(trial):
        params = suggest(trial, p1['space'])
        stem_cfg = stem_cfg_from_params(params)
        info = check_stem(stem_cfg, args.max_stem_cost_ratio, base_cfg)
        for k, v in info.items():
            trial.set_user_attr(k, v)
        if info['over_budget']:
            trial.set_user_attr('fail_reason', f'stem cost > {args.max_stem_cost_ratio}x SPT')
            raise optuna.TrialPruned(f"custo do stem: params {info['stem_params_ratio']:.2f}x, "
                                     f"FLOPs {info['stem_flops_ratio']:.2f}x o SPT")

        tdir = os.path.join(out_root, f'trial_{trial.number:04d}')
        cfg = {**base_cfg, 'stem_cfg': stem_cfg, 'stem_lr_mult': params['stem_lr_mult'],
               'epochs': args.search_epochs, 'eval_split': 'val', 'split_indices': split,
               'monitor': 'acc1', 'save_ckpt': 'best', 'out_dir': tdir}
        save_json({'trial': trial.number, 'study': args.study_name, 'params': params,
                   'config': {k: v for k, v in cfg.items() if k != 'split_indices'},
                   'split_file': p1['split_file']}, os.path.join(tdir, 'config.json'))

        def report(epoch, ev):
            if not args.multiobjective:
                trial.report(ev['acc1'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f'MedianPruner na época {epoch + 1}')

        result = run_guarded(trial, lambda: engine.train(cfg, epoch_callback=report))
        trial_metrics_attrs(trial, result)
        save_json({k: v for k, v in result.items() if k != 'history'}, os.path.join(tdir, 'result.json'))
        acc = result['final']['acc1']  # métrica da última época
        return (acc, info['stem_flops']) if args.multiobjective else acc

    optimize(study, objective, args.n_trials)

    if args.multiobjective:
        front = [{'number': t.number, 'values': t.values, 'params': t.params,
                  'stem_flops': t.user_attrs.get('stem_flops'), 'stem_params': t.user_attrs.get('stem_params')}
                 for t in study.best_trials]
        save_json(front, os.path.join(out_root, 'pareto_front.json'))
        print(f'[fase 1] fronteira de Pareto ({len(front)} pontos) -> {out_root}/pareto_front.json')
    else:
        try:
            print(f'[fase 1] melhor trial #{study.best_trial.number}: acc={study.best_value:.2f} '
                  f'params={study.best_trial.params}')
        except ValueError:
            print('[fase 1] nenhum trial COMPLETE ainda.')


if __name__ == '__main__':
    main()
