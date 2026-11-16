# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.

    `weight` (opcional): tensor [n_classes] com pesos por classe. Cada amostra é
    ponderada pelo peso da sua classe verdadeira e a média é normalizada pela
    soma dos pesos (mesma convenção do nn.CrossEntropyLoss com weight).
    Com weight=None o comportamento é idêntico ao original.
    """
    def __init__(self, smoothing=0.1, weight=None):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        losses = self._compute_losses(x, target)
        if self.weight is None:
            return losses.mean()
        w = self.weight[target]
        return (losses * w).sum() / w.sum()
