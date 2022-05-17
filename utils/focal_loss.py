from torch import nn
import torch

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 5.0,
        weight=None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inp: torch.Tensor, targ: torch.Tensor):
        ce_loss = F.cross_entropy(
            inp,
            targ,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
