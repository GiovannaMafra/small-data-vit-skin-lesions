"""
Loop de treino/avaliação reutilizável: train(config) -> métricas.

O laço de treino (CutMix/Mixup/RA/AdamW/warmup+cosine) é o mesmo do main.py
original; o main.py agora chama train_one_epoch/evaluate daqui.
"""
import csv
import gc
import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from models.create_model import create_model
from utils.cli import init_parser
from utils.cost import model_cost, stem_cost, throughput
from utils.dataloader import datainfo, build_train_transform, build_eval_transform, build_datasets
from utils.losses import LabelSmoothingCrossEntropy
from utils.mix import cutmix_data, mixup_data, mixup_criterion
from utils.print_progress import progress_bar
from utils.sampler import RASampler
from utils.scheduler import build_scheduler
from utils.training_functions import accuracy

logger = logging.getLogger('engine')
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Nome no pipeline -> flags do main.py
MODEL_FLAGS = {
    'vit':    dict(model='vit', is_SPT=False, is_LSA=False, is_CL=False),
    's_vit':  dict(model='vit', is_SPT=True,  is_LSA=False, is_CL=False),  # ablação: só SPT
    'l_vit':  dict(model='vit', is_SPT=False, is_LSA=True,  is_CL=False),  # ablação: só LSA
    'sl_vit': dict(model='vit', is_SPT=True,  is_LSA=True,  is_CL=False),
    'cl_vit': dict(model='vit', is_SPT=False, is_LSA=False, is_CL=True),   # is_CL ativa o LSA
}
DATASETS = {'cifar100': 'CIFAR100', 'tiny_imagenet': 'T-IMNET', 'dermamnist': 'DermaMNIST'}
DERMA_CLASSES = ['actinic_keratoses', 'basal_cell_carcinoma', 'benign_keratosis', 'dermatofibroma',
                 'melanoma', 'melanocytic_nevi', 'vascular_lesions']
MELANOMA_IDX = 4


class NonFiniteLossError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def set_seed(seed):
    # Igual ao main.py
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_args(cfg):
    """Namespace com os padrões do main.py (receita do artigo) sobrescritos por cfg."""
    args = init_parser().parse_args([])
    for k, v in MODEL_FLAGS[cfg.get('model', 'vit')].items():
        setattr(args, k, v)
    args.dataset = DATASETS[cfg.get('dataset', 'cifar100')]
    passthrough = {k: v for k, v in cfg.items() if k not in ('model', 'dataset')}
    for k, v in passthrough.items():
        setattr(args, k, v)
    args.stem_cfg = cfg.get('stem_cfg') if args.is_CL else None
    if args.warmup >= args.epochs:  # só em runs curtos (smoke test); 30/100 épocas mantêm warmup=10
        args.warmup = max(0, args.epochs // 10)
    return args


def class_weights(train_targets, n_classes, mode):
    if mode in (None, 'none'):
        return None
    counts = np.bincount(np.asarray(train_targets), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    if mode == 'sqrt_inverse_freq':
        w = np.sqrt(w)
    elif mode != 'inverse_freq':
        raise ValueError(f'class_weighting inválido: {mode}')
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def build_criterion(args, weight=None):
    if args.ls:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing, weight=weight)
    return nn.CrossEntropyLoss(weight=weight)


def build_optimizer(model, args):
    mult = float(getattr(args, 'stem_lr_mult', 1.0) or 1.0)
    if args.is_CL and mult != 1.0:
        stem_ids = {id(p) for p in model.to_patch_embedding.parameters()}
        rest = [p for p in model.parameters() if id(p) not in stem_ids]
        stem = [p for p in model.parameters() if id(p) in stem_ids]
        params = [{'params': rest, 'lr_mult': 1.0}, {'params': stem, 'lr_mult': mult}]
    else:
        params = model.parameters()
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def _limit(ds, n, seed, targets=None):
    """Subset aleatório de n amostras (smoke test). Devolve (ds, targets) com os rótulos alinhados."""
    if n is None or n <= 0 or n >= len(ds):
        return ds, targets
    idx = sorted(np.random.RandomState(seed).permutation(len(ds))[:n].tolist())
    return Subset(ds, idx), ([targets[i] for i in idx] if targets is not None else None)


# ----------------------------------------------------------------------------
# Laço de treino (mesmo do main.py original)
# ----------------------------------------------------------------------------

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, args, device, progress=False):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    lr = optimizer.param_groups[0]["lr"]

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Cutmix only
        if args.cm and not args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                output = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                output = model(images)
                loss = criterion(output, target)

        # Mixup only
        elif not args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                images, y_a, y_b, lam = mixup_data(images, target, args)
                output = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                output = model(images)
                loss = criterion(output, target)

        # Both Cutmix and Mixup
        elif args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                switching_prob = np.random.rand(1)
                # Cutmix
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(images)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                # Mixup
                else:
                    images, y_a, y_b, lam = mixup_data(images, target, args)
                    output = model(images)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                output = model(images)
                loss = criterion(output, target)

        # No Mix
        else:
            output = model(images)
            loss = criterion(output, target)

        if not torch.isfinite(loss):
            raise NonFiniteLossError(f'loss não finita na época {epoch + 1}, iteração {i}')

        acc = accuracy(output, target, (1,))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if progress and args.print_freq >= 0 and i % args.print_freq == 0:
            progress_bar(i, len(train_loader), f'[Epoch {epoch+1}/{args.epochs}][T][{i}]   Loss: {loss_val / n:.4e}   Top-1: {acc1_val / n:6.2f}   LR: {lr:.7f}' + ' ' * 10)

    return {'loss': loss_val / max(n, 1), 'acc1': acc1_val / max(n, 1), 'lr': lr}


def _macro_auc_ovr(y_true, y_prob, n_classes):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(n_classes):
        pos = (y_true == c)
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos.astype(int), y_prob[:, c]))
    return float(np.mean(aucs)) if aucs else float('nan')


@torch.no_grad()
def evaluate(val_loader, model, criterion, args, device, n_classes=None, full=True, with_cm=False,
             progress=False, epoch=0, lr=0.0):
    """
    Métricas na base de avaliação. Básicas: loss, acc1 (%), acc5 (%).
    full=True: + f1_macro, auc_macro_ovr, balanced_acc, recall_per_class; with_cm: + confusion_matrix.
    n_classes=None: inferido da saída do modelo.
    """
    model.eval()
    loss_val, acc1_val, acc5_val = 0.0, 0.0, 0.0
    n = 0
    probs, targets = [], []
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        if n_classes is None:
            n_classes = output.shape[1]
        topk = (1, min(5, n_classes))
        loss = criterion(output, target)
        acc = accuracy(output, target, topk)
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc[0][0] * images.size(0))
        acc5_val += float(acc[1][0] * images.size(0))
        if full:
            probs.append(torch.softmax(output.float(), dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
        if progress and args.print_freq >= 0 and i % args.print_freq == 0:
            progress_bar(i, len(val_loader), f'[Epoch {epoch+1}][V][{i}]   Loss: {loss_val / n:.4e}   Top-1: {acc1_val / n:6.2f}   LR: {lr:.6f}')

    m = {'loss': loss_val / max(n, 1), 'acc1': acc1_val / max(n, 1), 'acc5': acc5_val / max(n, 1)}
    if full:
        from sklearn.metrics import f1_score, recall_score, confusion_matrix, balanced_accuracy_score
        y_prob = np.concatenate(probs)
        y_true = np.concatenate(targets).ravel()
        y_pred = y_prob.argmax(1)
        labels = np.arange(n_classes)
        m['f1_macro'] = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
        m['balanced_acc'] = float(balanced_accuracy_score(y_true, y_pred))
        m['auc_macro_ovr'] = _macro_auc_ovr(y_true, y_prob, n_classes)
        m['recall_per_class'] = [float(r) for r in recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)]
        if with_cm:
            m['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return m


# ----------------------------------------------------------------------------
# train(config) -> métricas
# ----------------------------------------------------------------------------

def _scalar_row(epoch, tr, ev, is_derma=False):
    row = {'epoch': epoch + 1, 'lr': tr['lr'], 'train_loss': tr['loss'], 'train_acc1': tr['acc1']}
    for k, v in ev.items():
        if isinstance(v, (int, float)):
            row[f'eval_{k}'] = v
    if is_derma:
        row['eval_recall_melanoma'] = ev['recall_per_class'][MELANOMA_IDX]
    return row


def _write_history(out_dir, history):
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=1)
    keys = list(history[0].keys())
    with open(os.path.join(out_dir, 'history.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def train(cfg, epoch_callback=None):
    """
    Treina um modelo segundo `cfg` e devolve as métricas.

    Chaves principais de cfg (as demais são passadas ao Namespace do main.py):
      model: vit | sl_vit | cl_vit         dataset: cifar100 | tiny_imagenet | dermamnist
      eval_split: val | test               split_indices: dict (CIFAR-100 val)
      stem_cfg: dict (CL-ViT)              stem_lr_mult: float
      class_weighting: none | inverse_freq | sqrt_inverse_freq
      out_dir: onde salvar history.csv/json e checkpoint
      save_ckpt: best | last | none        monitor: métrica para o 'best' (ex. acc1, f1_macro)
      max_train_samples / max_eval_samples: subset (smoke test)
      device: cuda | cpu (padrão: cuda se disponível)
    epoch_callback(epoch, eval_metrics): chamado a cada época (pode levantar optuna.TrialPruned).
    """
    args = make_args(cfg)
    set_seed(args.seed)
    device = torch.device(cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = cfg.get('out_dir')
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    eval_split = cfg.get('eval_split', 'test')
    monitor = cfg.get('monitor', 'acc1')
    save_ckpt = cfg.get('save_ckpt', 'best')

    data_info = datainfo(logger, args)
    n_classes, img_size = data_info['n_classes'], data_info['img_size']
    is_derma = args.dataset == 'DermaMNIST'
    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]
    train_tf = build_train_transform(args, data_info, normalize)
    eval_tf = build_eval_transform(data_info, normalize, square=(args.dataset == 'DermaMNIST'))
    train_ds, eval_ds, train_targets = build_datasets(args, data_info, train_tf, eval_tf, eval_split,
                                                      cfg.get('split_indices'))
    train_ds, train_targets = _limit(train_ds, cfg.get('max_train_samples'), args.seed, train_targets)
    eval_ds, _ = _limit(eval_ds, cfg.get('max_eval_samples'), args.seed)

    pin = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds, num_workers=args.workers, pin_memory=pin,
        batch_sampler=RASampler(len(train_ds), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin,
                             num_workers=args.workers)

    model = create_model(img_size, n_classes, args).to(device)
    weight = class_weights(train_targets, n_classes, cfg.get('class_weighting', 'none'))
    criterion = build_criterion(args, weight.to(device) if weight is not None else None).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    cost = model_cost(model, img_size)
    s_cost = stem_cost(model.to_patch_embedding, img_size)
    info = {'model_params': cost['params'], 'model_flops': cost['flops'],
            'stem_params': s_cost['params'], 'stem_flops': s_cost['flops'], 'n_tokens': s_cost['n_tokens'],
            'n_train': len(train_ds), 'n_eval': len(eval_ds), 'eval_split': eval_split,
            'class_weights': weight.tolist() if weight is not None else None}
    logger.info(f"[engine] {cfg.get('model')} {args.dataset} | params={cost['params']:,} "
                f"flops={cost['flops']:,} | stem params={s_cost['params']:,} flops={s_cost['flops']:,} "
                f"tokens={s_cost['n_tokens']} | train={len(train_ds)} eval={len(eval_ds)} ({eval_split}) | device={device}")

    history, best, best_metrics = [], -math.inf, None
    ev = None
    for epoch in range(args.epochs):
        last = epoch == args.epochs - 1
        tr = train_one_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, args, device)
        ev = evaluate(eval_loader, model, criterion, args, device, n_classes, full=True, with_cm=last)
        row = _scalar_row(epoch, tr, ev, is_derma)
        history.append(row)
        logger.info(f"[epoch {epoch+1}/{args.epochs}] train_loss={tr['loss']:.4f} eval_acc1={ev['acc1']:.2f} "
                    f"f1={ev['f1_macro']:.4f} auc={ev['auc_macro_ovr']:.4f} lr={tr['lr']:.2e}")
        if out_dir:
            _write_history(out_dir, history)
        score = ev.get(monitor, ev['acc1'])
        if score > best:
            best, best_metrics = score, dict(row)
            if out_dir and save_ckpt == 'best':
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'metrics': row,
                            'config': _jsonable(cfg)}, os.path.join(out_dir, 'best.pth'))
        if epoch_callback is not None:
            epoch_callback(epoch, ev)

    # Métricas finais = avaliação da ÚLTIMA época (já inclui a matriz de confusão)
    final = dict(ev)
    if is_derma:
        final['recall_melanoma'] = final['recall_per_class'][MELANOMA_IDX]
    if out_dir and save_ckpt == 'last':
        torch.save({'model_state_dict': model.state_dict(), 'epoch': args.epochs - 1, 'metrics': final,
                    'config': _jsonable(cfg)}, os.path.join(out_dir, 'last.pth'))
    info['throughput_img_s'] = throughput(model, img_size, device)

    result = {'final': final, 'best': best_metrics, 'monitor': monitor, 'history': history, **info}
    del model, optimizer, scheduler
    free_memory()
    return result


def _jsonable(cfg):
    return {k: v for k, v in cfg.items() if k != 'split_indices'}


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()
