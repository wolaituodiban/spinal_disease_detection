import torch


class PosBCELoss:
    def __init__(self, max_dist=8):
        self.max_dist = max_dist

    def __call__(self, pred: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor):
        mask = mask.to(pred.device)
        label = dist.to(pred.device) < self.max_dist
        pred = torch.masked_select(pred, mask)
        label = torch.masked_select(label, mask).to(pred.dtype)
        pos_ratio = label.mean()
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=1/pos_ratio)
        return loss(pred, label)


class NullLoss:
    def __call__(self, x, *args):
        return x.mean()
