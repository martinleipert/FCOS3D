import torch
from torch import nn


# Sigmoid Focal Loss
# from https://github.com/rosinality/fcos-pytorch/blob/master/loss.py


def clip_sigmoid(input):
    out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)

    return out


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        eps = 1e-6
        if alpha < eps or alpha > (1 - eps):
            Warning("alpha must be a value between 0 and 1 weighting the influence of background")

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        loss = (
                -(t == class_ids).float() * alpha * term1
                - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()
