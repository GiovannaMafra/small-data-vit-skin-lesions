"""
Fase 3: treinos finais (sem Optuna). Treino no treino completo, avaliação UMA vez no teste,
métricas da ÚLTIMA época.

  CIFAR-100     : treino completo (50k = treino + os 10% de val da Fase 1) -> test
  Tiny-ImageNet : train -> val oficial (= teste, como no artigo). Sem busca; stem transferido do CIFAR-100.
  DermaMNIST    : train oficial -> test; hiperparâmetros da Fase 2 de cada modelo.

--params-from paper           receita do artigo (padrão para CIFAR-100 e Tiny-ImageNet)
--params-from study:<nome>    melhores params do study (padrão no Derma: study:phase2_derma_{model})

Ex.: python final_train.py --model cl_vit --dataset tiny_imagenet --seed 1
"""
import os

import optuna

from utils import engine
from utils.pipeline import (DEFAULT_STEM_FINAL, common_parser, get_storage, load_final_stem, load_yaml,
                            recipe_config, save_json)


def parse():
    p = common_parser(__doc__)
    p.add_argument('--model', required=True, choices=['vit', 'sl_vit', 'cl_vit'])
    p.add_argument('--dataset', required=True, choices=['cifar100', 'tiny_imagenet', 'dermamnist'])
    p.add_argument('--params-from', default=None, help='paper | study:<nome>')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--stem-config', default=DEFAULT_STEM_FINAL)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def main():
    args = parse()
    spaces = load_yaml(args.search_spaces)
    params_from = args.params_from or (f'study:phase2_derma_{args.model}' if args.dataset == 'dermamnist' else 'paper')
    out_dir = os.path.join(args.results_dir, 'final', args.dataset, args.model, f'seed{args.seed}')
    final_path = os.path.join(out_dir, 'final_metrics.json')
    if os.path.exists(final_path) and not args.overwrite:
        print(f'[final] {final_path} já existe; pulando (use --overwrite).')
        return

    cfg = recipe_config(spaces, args.model, args.dataset, args)
    cfg.update(epochs=args.epochs, eval_split='test', monitor='acc1', save_ckpt='last', out_dir=out_dir)
    if args.dataset == 'dermamnist':
        cfg['monitor'] = 'f1_macro'

    hp = {}
    if params_from.startswith('study:'):
        name = params_from.split(':', 1)[1]
        study = optuna.load_study(study_name=name, storage=get_storage(args.storage_dir, name))
        best = study.best_trial
        hp = dict(best.params)
        cfg.update(lr=hp['lr'], smoothing=hp['label_smoothing'], class_weighting=hp['class_weighting'])
        hp['_source'] = f'{name}#trial{best.number}'
    elif params_from != 'paper':
        raise ValueError(f'--params-from inválido: {params_from}')

    if args.model == 'cl_vit':
        stem_cfg, stem_lr_mult = load_final_stem(args.stem_config)
        cfg.update(stem_cfg=stem_cfg, stem_lr_mult=stem_lr_mult)

    save_json({'model': args.model, 'dataset': args.dataset, 'seed': args.seed, 'params_from': params_from,
               'hparams': hp, 'config': cfg}, os.path.join(out_dir, 'config.json'))
    result = engine.train(cfg)

    out = {'model': args.model, 'dataset': args.dataset, 'seed': args.seed, 'params_from': params_from,
           'hparams': hp, 'epochs': args.epochs, 'metric_epoch': 'last',
           'test': result['final'],  # métricas da última época, avaliadas uma vez no teste
           # melhor época no teste (pelo monitor: acc1, ou f1_macro no Derma). Serve só para comparar
           # com as reproduções e com o artigo, que reportam a best acc; é otimista (escolhida no teste).
           'test_best': result['best'], 'best_monitor': result['monitor'],
           **{k: result[k] for k in ('model_params', 'model_flops', 'stem_params', 'stem_flops', 'n_tokens',
                                     'throughput_img_s', 'n_train', 'n_eval', 'eval_split', 'class_weights')}}
    save_json(out, final_path)
    print(f"[final] {args.model}/{args.dataset}/seed{args.seed}: acc={result['final']['acc1']:.2f} "
          f"f1={result['final']['f1_macro']:.4f} auc={result['final']['auc_macro_ovr']:.4f} -> {final_path}")


if __name__ == '__main__':
    main()
