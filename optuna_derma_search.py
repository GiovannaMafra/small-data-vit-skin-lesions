"""
Fase 2: ajuste para desbalanceamento no DermaMNIST (val oficial), igual para os 3 modelos.

Mesmo espaço, mesmo nº de trials e mesmo seed do sampler para vit, sl_vit e cl_vit.
O CL-ViT usa o stem fixo de configs/cl_vit_stem_final.yaml.
Objetivo: F1 macro na validação (última época).

Ex.: python optuna_derma_search.py --model sl_vit
"""
import os

import optuna

from utils import engine
from utils.pipeline import (DEFAULT_STEM_FINAL, add_optuna_args, common_parser, load_final_stem, load_yaml,
                            make_study, optimize, recipe_config, run_guarded, save_json, suggest,
                            trial_metrics_attrs)


def parse():
    p = common_parser(__doc__)
    add_optuna_args(p, n_trials=20, epochs=30)
    p.add_argument('--model', required=True, choices=['vit', 'sl_vit', 'cl_vit'])
    p.add_argument('--stem-config', default=DEFAULT_STEM_FINAL)
    p.add_argument('--study-name', default=None, help='padrão: phase2_derma_{model}')
    return p.parse_args()


def main():
    args = parse()
    spaces = load_yaml(args.search_spaces)
    p2 = spaces['phase2']
    study_name = args.study_name or f'phase2_derma_{args.model}'
    out_root = os.path.join(args.results_dir, 'phase2', args.model)
    study, _ = make_study(args, study_name, ['maximize'])

    base_cfg = recipe_config(spaces, args.model, p2['dataset'], args)
    if args.model == 'cl_vit':
        stem_cfg, stem_lr_mult = load_final_stem(args.stem_config)
        base_cfg.update(stem_cfg=stem_cfg, stem_lr_mult=stem_lr_mult)
        study.set_user_attr('stem_cfg', stem_cfg)
        study.set_user_attr('stem_lr_mult', stem_lr_mult)

    def objective(trial):
        params = suggest(trial, p2['space'])
        tdir = os.path.join(out_root, f'trial_{trial.number:04d}')
        cfg = {**base_cfg, 'lr': params['lr'], 'smoothing': params['label_smoothing'],
               'class_weighting': params['class_weighting'], 'epochs': args.search_epochs,
               'eval_split': 'val', 'monitor': 'f1_macro', 'save_ckpt': 'best', 'out_dir': tdir}
        save_json({'trial': trial.number, 'study': study_name, 'params': params, 'config': cfg},
                  os.path.join(tdir, 'config.json'))

        def report(epoch, ev):
            trial.report(ev['f1_macro'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f'MedianPruner na época {epoch + 1}')

        result = run_guarded(trial, lambda: engine.train(cfg, epoch_callback=report))
        trial_metrics_attrs(trial, result)
        save_json({k: v for k, v in result.items() if k != 'history'}, os.path.join(tdir, 'result.json'))
        return result['final']['f1_macro']  # última época

    optimize(study, objective, args.n_trials)
    try:
        b = study.best_trial
        print(f'[fase 2] {study_name}: melhor trial #{b.number} F1={b.value:.4f} '
              f'AUC={b.user_attrs.get("final_auc_macro_ovr")} recall_mel={b.user_attrs.get("final_recall_melanoma")} '
              f'params={b.params}')
    except ValueError:
        print(f'[fase 2] {study_name}: nenhum trial COMPLETE ainda.')


if __name__ == '__main__':
    main()
