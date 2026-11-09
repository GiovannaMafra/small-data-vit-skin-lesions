import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import optuna

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import roc_auc_score, f1_score

from main import init_parser
from models.create_model import create_model

from utils.dataloader import datainfo, dataload
from utils.losses import LabelSmoothingCrossEntropy
from utils.scheduler import build_scheduler
from utils.mix import (
    cutmix_data,
    mixup_data,
    mixup_criterion
)
from utils.sampler import RASampler


# ==========================================
# CONFIGURAÇÕES
# ==========================================

SEED = 42
EPOCHS = 100
DATA_PATH = "./data"
STUDY_NAME = "clvit_derma"
DATABASE = "sqlite:///optuna_clvit.db"


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==========================================
# CONSTRUÇÃO DO EXPERIMENTO
# ==========================================

def build(cfg, seed):

    set_seed(seed)

    parser = init_parser()
    args = parser.parse_args([])

    # Dataset
    args.dataset = "DermaMNIST"
    args.data_path = DATA_PATH

    # Modelo
    args.model = "vit"
    args.is_CL = True
    args.is_LSA = False
    args.is_SPT = False

    # Hiperparâmetros
    args.epochs = EPOCHS
    args.batch_size = cfg["bs"]
    args.lr = cfg["lr"]
    args.weight_decay = cfg["wd"]
    args.warmup = cfg["warmup"]
    args.sd = cfg["drop_path"]
    args.smoothing = cfg["smooth"]
    args.mix_prob = cfg["mix_prob"]

    # Configurações fixas
    args.gpu = 0
    args.workers = 4
    args.no_cuda = False

    args.ra = 1
    args.beta = 1.0
    args.alpha = 1.0

    args.cm = True
    args.mu = True

    device = torch.device("cuda:0")

    # Informações do dataset
    logger = logging.getLogger("optuna")
    data_info = datainfo(logger, args)

    # Transformações
    normalize = [
        transforms.Normalize(
            mean=data_info["stat"][0],
            std=data_info["stat"][1]
        )
    ]

    augmentations = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        *normalize
    ])

    # Construção dos datasets
    train_dataset, val_dataset = dataload(
        args,
        augmentations,
        normalize,
        data_info
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # Modelo
    model = create_model(
        data_info["img_size"],
        data_info["n_classes"],
        args
    ).to(device)

    # Loss
    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.smoothing
    ).to(device)

    # Otimizador
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = build_scheduler(
        args,
        optimizer,
        len(train_loader)
    )

    return (
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        args,
        device
    )


# ==========================================
# TREINAMENTO
# ==========================================

def train_one_epoch(
    model,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    args,
    device
):

    model.train()

    total_loss = 0.0

    for images, target in train_loader:

        images = images.to(device, non_blocking=True)

        target = target.to(
            device,
            non_blocking=True
        ).long().view(-1)

        optimizer.zero_grad()

        r = np.random.rand()

        # Mixup e Cutmix
        if r < args.mix_prob:

            switching_prob = np.random.rand()

            if switching_prob < 0.5:

                slicing_idx, y_a, y_b, lam, sliced = (
                    cutmix_data(images, target, args)
                )

                images[
                    :,
                    :,
                    slicing_idx[0]:slicing_idx[2],
                    slicing_idx[1]:slicing_idx[3]
                ] = sliced

                output = model(images)

                loss = mixup_criterion(
                    criterion,
                    output,
                    y_a,
                    y_b,
                    lam
                )

            else:

                images, y_a, y_b, lam = mixup_data(
                    images,
                    target,
                    args
                )

                output = model(images)

                loss = mixup_criterion(
                    criterion,
                    output,
                    y_a,
                    y_b,
                    lam
                )

        else:

            output = model(images)
            loss = criterion(output, target)

        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ==========================================
# VALIDAÇÃO
# ==========================================

def evaluate(model, val_loader, device):

    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():

        for images, target in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            target = target.view(-1)

            output = model(images)

            probs = torch.softmax(output, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(target.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets).ravel()

    y_pred = np.argmax(y_prob, axis=1)

    auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=np.arange(7)
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    return auc, f1


# ==========================================
# OPTUNA
# ==========================================

def objective(trial):

    cfg = {

        "base_lr": trial.suggest_float(
            "base_lr",
            3e-4,
            5e-3,
            log=True
        ),

        "wd": trial.suggest_float(
            "wd",
            0.01,
            0.1,
            log=True
        ),

        "bs": trial.suggest_categorical(
            "bs",
            [64, 128, 256]
        ),

        "warmup": trial.suggest_int(
            "warmup",
            5,
            15
        ),

        "drop_path": trial.suggest_float(
            "drop_path",
            0.0,
            0.2
        ),

        "smooth": trial.suggest_float(
            "smooth",
            0.0,
            0.1
        ),

        "mix_prob": trial.suggest_float(
            "mix_prob",
            0.0,
            1.0
        )
    }

    cfg["lr"] = (
        cfg["base_lr"] * cfg["bs"] / 128
    )

    (
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        args,
        device
    ) = build(cfg, SEED)

    auc_history = []
    f1_history = []

    try:

        for epoch in range(EPOCHS):

            train_one_epoch(
                model,
                optimizer,
                scheduler,
                criterion,
                train_loader,
                args,
                device
            )

            auc, f1 = evaluate(
                model,
                val_loader,
                device
            )

            auc_history.append(auc)
            f1_history.append(f1)

            trial.report(auc, epoch)

            trial.set_user_attr(
                f"f1_epoch_{epoch}",
                float(f1)
            )

            if trial.should_prune():

                raise optuna.TrialPruned()

        final_auc = float(
            np.mean(auc_history[-5:])
        )

        final_f1 = float(
            np.mean(f1_history[-5:])
        )

        trial.set_user_attr(
            "final_f1",
            final_f1
        )

        return final_auc

    finally:

        del model
        del optimizer
        del scheduler

        torch.cuda.empty_cache()


# ==========================================
# EXECUÇÃO
# ==========================================

if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    study = optuna.create_study(

        direction="maximize",

        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),

        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8,
            n_warmup_steps=30
        ),

        storage=DATABASE,

        study_name=STUDY_NAME,

        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=40
    )

    print("\n========== RESULTADO OPTUNA ==========")

    print("Melhor AUC:", study.best_value)

    print("Melhores parâmetros:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    print(
        "F1 médio:",
        study.best_trial.user_attrs.get("final_f1")
    )

    study.trials_dataframe().to_csv(
        "optuna_results.csv",
        index=False
    )