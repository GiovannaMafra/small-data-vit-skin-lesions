"""Split estratificado treino/validação com índices salvos em arquivo (reprodutível)."""
import json
import os

import numpy as np


def stratified_split(targets, val_frac=0.1, seed=0):
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def get_or_create_split(path, targets, val_frac=0.1, seed=0):
    """Carrega os índices de `path`; se não existir, cria (escrita atômica, segura entre workers)."""
    if os.path.exists(path):
        with open(path) as f:
            split = json.load(f)
        assert split['seed'] == seed and abs(split['val_frac'] - val_frac) < 1e-9, \
            f'{path} foi criado com outro seed/fração; apague-o ou use outro caminho'
        assert len(split['train_idx']) + len(split['val_idx']) == len(targets)
        return split
    train_idx, val_idx = stratified_split(targets, val_frac, seed)
    split = {'seed': seed, 'val_frac': val_frac, 'n_total': len(targets),
             'train_idx': train_idx, 'val_idx': val_idx}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(split, f)
    os.replace(tmp, path)
    return split
