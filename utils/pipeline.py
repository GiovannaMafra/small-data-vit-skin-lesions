"""Utilitários compartilhados pelos scripts do pipeline Optuna (Fases 1-3)."""
import argparse
import json
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPACES = os.path.join(REPO, 'configs', 'search_spaces.yaml')
DEFAULT_STEM_FINAL = os.path.join(REPO, 'configs', 'cl_vit_stem_final.yaml')


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Optuna storage (JournalStorage em arquivo; seguro para vários workers em NFS)
# ---------------------------------------------------------------------------

def get_storage(storage_dir, study_name):
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    os.makedirs(storage_dir, exist_ok=True)
    return JournalStorage(JournalFileBackend(os.path.join(storage_dir, f'{study_name}.log')))


def worker_id(cli_value=None):
    if cli_value is not None:
        return int(cli_value)
    return int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))


def count_finished(study):
    import optuna
    return sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
               for t in study.get_trials(deepcopy=False))


# ---------------------------------------------------------------------------
# Config base (receita do artigo)
# ---------------------------------------------------------------------------

def recipe_config(spaces, model, dataset, args):
    """Config do engine com a receita do artigo para (modelo, base)."""
    cfg = dict(spaces['recipe'].get('overrides') or {})
    cfg.update(model=model, dataset=dataset, lr=float(spaces['recipe']['lr'][model]),
               data_path=args.data_path, workers=args.workers, seed=args.seed)
    if getattr(args, 'device', None):
        cfg['device'] = args.device
    if getattr(args, 'max_train_samples', None):
        cfg['max_train_samples'] = args.max_train_samples
    if getattr(args, 'max_eval_samples', None):
        cfg['max_eval_samples'] = args.max_eval_samples
    if getattr(args, 'batch_size', None):
        cfg['batch_size'] = args.batch_size
    return cfg


def stem_cfg_from_params(p):
    return {'channels': int(p['stem_channels']), 'growth': p['stem_channel_growth'],
            'extra_conv': bool(p['stem_extra_conv']), 'final_proj': p['stem_final_proj']}


def load_final_stem(path=DEFAULT_STEM_FINAL):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} não existe: rode a Fase 1 + confirm_top_stems.py antes.')
    d = load_yaml(path)
    return d['stem_cfg'], float(d.get('stem_lr_mult', 1.0))


def common_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data-path', dest='data_path', default=os.environ.get('DATA_PATH', './data'))
    p.add_argument('--storage-dir', default=os.environ.get('OPTUNA_DIR', './optuna_storage'),
                   help='diretório compartilhado com os journals do Optuna')
    p.add_argument('--results-dir', default=os.environ.get('RESULTS_DIR', './results'))
    p.add_argument('--search-spaces', default=DEFAULT_SPACES)
    p.add_argument('--workers', type=int, default=4, help='workers do DataLoader')
    p.add_argument('--seed', type=int, default=0, help='seed do treino')
    p.add_argument('--device', default=None, help='cuda | cpu (padrão: cuda se disponível)')
    p.add_argument('--batch-size', dest='batch_size', type=int, default=None, help='padrão: o do main.py (128)')
    # smoke test
    p.add_argument('--max-train-samples', type=int, default=None, help='subset do treino (smoke test)')
    p.add_argument('--max-eval-samples', type=int, default=None, help='subset da avaliação (smoke test)')
    return p


def add_optuna_args(p, n_trials, epochs):
    p.add_argument('--n-trials', type=int, default=n_trials, help='nº TOTAL de trials do study (todos os workers)')
    p.add_argument('--search-epochs', type=int, default=epochs)
    p.add_argument('--sampler-seed', type=int, default=0,
                   help='seed base do sampler; o worker i usa sampler_seed + i (evita trials duplicados)')
    p.add_argument('--worker-id', type=int, default=None, help='padrão: $SLURM_ARRAY_TASK_ID ou 0')
    p.add_argument('--pruner-startup-trials', type=int, default=5)
    p.add_argument('--pruner-warmup-epochs', type=int, default=10)
    return p


# ---------------------------------------------------------------------------
# Helpers de Optuna
# ---------------------------------------------------------------------------

class TrialOOM(Exception):
    """OOM na GPU: o trial vira FAIL (capturado em study.optimize) e o worker continua."""


def suggest(trial, space):
    """Lista -> categórico; dict(low, high, log) -> float."""
    out = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            out[name] = trial.suggest_categorical(name, spec)
        else:
            out[name] = trial.suggest_float(name, float(spec['low']), float(spec['high']),
                                            log=bool(spec.get('log', False)))
    return out


def run_guarded(trial, fn):
    """Executa fn(); NaN -> PRUNED, OOM -> FAIL, sem derrubar o worker."""
    import optuna
    from utils.engine import NonFiniteLossError, is_oom, free_memory
    try:
        return fn()
    except optuna.TrialPruned:
        raise
    except NonFiniteLossError as e:
        trial.set_user_attr('fail_reason', f'nan: {e}')
        raise optuna.TrialPruned(str(e))
    except RuntimeError as e:
        if is_oom(e):
            trial.set_user_attr('fail_reason', 'oom')
            free_memory()
            raise TrialOOM(str(e)[:500])
        raise
    finally:
        free_memory()


def make_study(args, study_name, directions, multiobjective=False):
    import optuna
    wid = worker_id(args.worker_id)
    seed = args.sampler_seed + wid
    if multiobjective:
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
        pruner = optuna.pruners.NopPruner()
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, constant_liar=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=args.pruner_startup_trials,
                                             n_warmup_steps=args.pruner_warmup_epochs)
    study = optuna.create_study(study_name=study_name, storage=get_storage(args.storage_dir, study_name),
                                directions=directions, sampler=sampler, pruner=pruner, load_if_exists=True)
    return study, wid


def optimize(study, objective, n_trials):
    """Roda até o study ter n_trials COMPLETE+PRUNED (somando todos os workers)."""
    import optuna
    done = count_finished(study)
    if done >= n_trials:
        print(f'[optuna] {study.study_name}: já tem {done}/{n_trials} trials terminados; nada a fazer.')
        return
    cb = optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,
                                                          optuna.trial.TrialState.PRUNED))
    # n_trials por worker é só um teto de segurança (ex.: OOMs em série)
    study.optimize(objective, n_trials=2 * n_trials, callbacks=[cb], catch=(TrialOOM,), gc_after_trial=True)


def trial_metrics_attrs(trial, result):
    """Registra no trial as métricas finais (última época) e custos."""
    fin = result['final']
    for k in ('acc1', 'acc5', 'f1_macro', 'auc_macro_ovr', 'balanced_acc', 'loss', 'recall_melanoma'):
        if k in fin:
            trial.set_user_attr(f'final_{k}', fin[k])
    trial.set_user_attr('recall_per_class', fin['recall_per_class'])
    if len(fin.get('confusion_matrix', [])) <= 10:  # não guarda matriz 100x100 no journal
        trial.set_user_attr('confusion_matrix', fin['confusion_matrix'])
    for k in ('model_params', 'model_flops', 'stem_params', 'stem_flops', 'n_tokens', 'throughput_img_s'):
        trial.set_user_attr(k, result[k])
    if result.get('best'):
        trial.set_user_attr('best_epoch', result['best']['epoch'])
