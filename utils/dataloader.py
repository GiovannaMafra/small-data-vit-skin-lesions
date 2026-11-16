import os
from colorama import Fore, Style
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image


def datainfo(logger, args):
    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR10')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32

    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        img_size = 32

    elif args.dataset == 'SVHN':
        print(Fore.YELLOW+'*'*80)
        logger.debug('SVHN')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        img_size = 32

    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('T-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64

    elif args.dataset == 'DermaMNIST':
        print(Fore.YELLOW + '*' * 80)
        logger.debug('DermaMNIST')
        print('*' * 80 + Style.RESET_ALL)
        n_classes = 7
        img_mean = (0.5, 0.5, 0.5)
        img_std = (0.5, 0.5, 0.5)
        img_size = 32

    data_info = dict()
    data_info['n_classes'] = n_classes
    data_info['stat'] = (img_mean, img_std)
    data_info['img_size'] = img_size

    return data_info


# ----------------------------------------------------------------------------
# Transformações (antes montadas dentro do main.py; a lógica é a mesma)
# ----------------------------------------------------------------------------

def build_train_transform(args, data_info, normalize, logger=None):
    """Augmentation do artigo: flip + crop(pad 4) + AutoAugment + RandomErasing."""
    augmentations = []

    # DermaMNIST: 28x28 -> 32x32 antes de qualquer augmentation (igual à validação)
    if args.dataset == 'DermaMNIST':
        augmentations += [transforms.Resize((data_info['img_size'], data_info['img_size']))]

    augmentations += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4)
    ]

    if args.aa == True:
        if 'CIFAR' in args.dataset:
            from utils.autoaug import CIFAR10Policy
            augmentations += [CIFAR10Policy()]
        elif 'SVHN' in args.dataset:
            from utils.autoaug import SVHNPolicy
            augmentations += [SVHNPolicy()]
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [ImageNetPolicy()]

    augmentations += [
        transforms.ToTensor(),
        *normalize]

    if args.re > 0:
        from utils.random_erasing import RandomErasing
        augmentations += [
            RandomErasing(probability=args.re, sh=args.re_sh, r1=args.re_r1, mean=data_info['stat'][0])
        ]

    return transforms.Compose(augmentations)


def build_eval_transform(data_info, normalize, square=False):
    size = (data_info['img_size'], data_info['img_size']) if square else data_info['img_size']
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize])


# ----------------------------------------------------------------------------
# Datasets auxiliares
# ----------------------------------------------------------------------------

def medmnist_target(y):
    """MedMNIST devolve o rótulo como array shape (1,); converte para int."""
    return int(y[0])


class TinyImageNetRawVal(Dataset):
    """Val oficial do Tiny-ImageNet no formato bruto (val/images + val_annotations.txt)."""
    def __init__(self, root, class_to_idx, transform=None):
        self.transform = transform
        self.samples = []
        with open(os.path.join(root, 'val_annotations.txt')) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                self.samples.append((os.path.join(root, 'images', parts[0]), class_to_idx[parts[1]]))
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _tiny_imagenet(data_path, train_tf, eval_tf):
    legacy = os.path.join(data_path, 'tiny_imagenet')
    raw = os.path.join(data_path, 'tiny-imagenet-200')
    if os.path.isdir(os.path.join(legacy, 'train')):
        # Layout original do repositório: val já reorganizado em pastas por classe
        train_dataset = datasets.ImageFolder(root=os.path.join(legacy, 'train'), transform=train_tf)
        val_dataset = datasets.ImageFolder(root=os.path.join(legacy, 'val'), transform=eval_tf)
    elif os.path.isdir(os.path.join(raw, 'train')):
        # Layout bruto do zip oficial (cs231n)
        train_dataset = datasets.ImageFolder(root=os.path.join(raw, 'train'), transform=train_tf)
        val_dataset = TinyImageNetRawVal(os.path.join(raw, 'val'), train_dataset.class_to_idx, transform=eval_tf)
    else:
        raise FileNotFoundError(f'Tiny-ImageNet não encontrado em {legacy} nem em {raw}')
    return train_dataset, val_dataset


def dataload(args, augmentations, normalize, data_info):
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=build_eval_transform(data_info, normalize))

    elif args.dataset == 'CIFAR100':

        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=build_eval_transform(data_info, normalize))

    elif args.dataset == 'SVHN':

        train_dataset = datasets.SVHN(
            root=args.data_path, split='train', download=True, transform=augmentations)
        val_dataset = datasets.SVHN(
            root=args.data_path, split='test', download=True, transform=build_eval_transform(data_info, normalize))

    elif args.dataset == 'T-IMNET':
        train_dataset, val_dataset = _tiny_imagenet(
            args.data_path, augmentations, build_eval_transform(data_info, normalize))

    elif args.dataset == 'DermaMNIST':
        train_dataset, val_dataset = _dermamnist(args.data_path, augmentations,
                                                 build_eval_transform(data_info, normalize, square=True), 'val')

    return train_dataset, val_dataset


def _dermamnist(data_path, train_tf, eval_tf, eval_split):
    from medmnist import DermaMNIST
    train_dataset = DermaMNIST(split='train', root=data_path, download=False,
                               transform=train_tf, target_transform=medmnist_target)
    eval_dataset = DermaMNIST(split=eval_split, root=data_path, download=False,
                              transform=eval_tf, target_transform=medmnist_target)
    return train_dataset, eval_dataset


def dataset_targets(ds):
    """Rótulos inteiros de um dataset (sem carregar imagens)."""
    if hasattr(ds, 'targets'):
        return [int(t) for t in ds.targets]
    if hasattr(ds, 'labels'):  # MedMNIST
        return [int(t) for t in ds.labels.reshape(-1)]
    raise AttributeError('dataset sem targets/labels')


def build_datasets(args, data_info, train_tf, eval_tf, eval_split, split_indices=None):
    """
    Datasets para o pipeline Optuna/treino final.

    eval_split:
      'val'  -> validação (CIFAR-100: 10% do treino via split_indices; Derma: val oficial)
      'test' -> teste (CIFAR-100: test; Derma: test; Tiny-ImageNet: val oficial, como no artigo)
    split_indices: dict(train_idx, val_idx) — obrigatório para CIFAR-100 com eval_split='val'.
    Retorna (train_dataset, eval_dataset, train_targets).
    """
    from torch.utils.data import Subset

    if args.dataset == 'CIFAR100':
        train_full = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_tf)
        if eval_split == 'val':
            assert split_indices is not None, 'CIFAR-100 val precisa dos índices do split'
            eval_base = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=eval_tf)
            targets = dataset_targets(train_full)
            train_ds = Subset(train_full, split_indices['train_idx'])
            eval_ds = Subset(eval_base, split_indices['val_idx'])
            train_targets = [targets[i] for i in split_indices['train_idx']]
            return train_ds, eval_ds, train_targets
        eval_ds = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=eval_tf)
        return train_full, eval_ds, dataset_targets(train_full)

    if args.dataset == 'T-IMNET':
        assert eval_split == 'test', 'Tiny-ImageNet: sem busca; o val oficial é usado como teste'
        train_ds, eval_ds = _tiny_imagenet(args.data_path, train_tf, eval_tf)
        return train_ds, eval_ds, dataset_targets(train_ds)

    if args.dataset == 'DermaMNIST':
        train_ds, eval_ds = _dermamnist(args.data_path, train_tf, eval_tf, eval_split)
        return train_ds, eval_ds, dataset_targets(train_ds)

    raise ValueError(f'dataset não suportado no pipeline: {args.dataset}')
