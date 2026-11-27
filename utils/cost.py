"""Custo (parâmetros, FLOPs, vazão) de modelos e stems. FLOPs via fvcore (1 FLOP = 1 MAC)."""
import time
import torch
import torch.nn as nn


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def _hook_macs(module, inp):
    """Fallback simples: MACs de Conv2d/Linear via hooks."""
    total = [0]

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total[0] += o.numel() // o.shape[0] * k

    def lin_hook(m, i, o):
        total[0] += o.numel() // o.shape[0] * m.in_features

    hs = []
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        module(inp)
    for h in hs:
        h.remove()
    return total[0]


def count_flops(module, inp):
    """FLOPs (MACs) para UMA imagem."""
    was_training = module.training
    module.eval()
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(module, inp)
        fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        flops = int(fca.total())
    except Exception:
        flops = int(_hook_macs(module, inp))
    finally:
        module.train(was_training)
    return flops


def stem_cost(stem, img_size, in_chans=3):
    """Parâmetros, FLOPs e nº de tokens de um módulo de tokenização (B, C, H, W) -> (B, N, D)."""
    device = next(stem.parameters()).device
    x = torch.zeros(1, in_chans, img_size, img_size, device=device)
    was_training = stem.training
    stem.eval()
    with torch.no_grad():
        n_tokens = int(stem(x).shape[1])
    stem.train(was_training)
    return {'params': count_params(stem), 'flops': count_flops(stem, x), 'n_tokens': n_tokens}


def spt_reference_cost(img_size, patch_size, dim=192):
    """Custo do SPT do SL-ViT (mesma instanciação de models/vit.py)."""
    from models.SPT import ShiftedPatchTokenization
    spt = ShiftedPatchTokenization(3, dim, patch_size, is_pe=True)
    return stem_cost(spt, img_size)


def model_cost(model, img_size):
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    return {'params': count_params(model), 'flops': count_flops(model, x)}


@torch.no_grad()
def throughput(model, img_size, device, batch_size=128, warmup=3, iters=10):
    """Imagens/s em inferência (eval, batch fixo, entrada aleatória)."""
    was_training = model.training
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    model.train(was_training)
    return batch_size * iters / dt
