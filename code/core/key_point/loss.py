import math
import torch


class NullLoss:
    def __call__(self, x, *args):
        return x.mean()


class KeyPointBCELoss:
    def __init__(self, max_dist=8):
        self.max_dist = max_dist

    def __call__(self, pred: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor):
        label = dist.to(pred.device)

        pred = pred[mask]
        label = label[mask]
        label = label < self.max_dist
        label = label.to(pred.dtype)

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=1/label.mean())
        return loss(pred, label)


class KeyPointBCELossV2:
    def __init__(self, lamb=8/math.log(2)):
        self.lamb = lamb

    def __call__(self, pred: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor):
        label = dist.to(pred.device)

        pred = pred[mask]
        label = label[mask]
        label = 1 / (label/self.lamb).exp()

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=1/label.mean())
        return loss(pred, label)
