import torch


class DisLoss:
    def __call__(self, pred, target, mask):
        target = target.to(device=pred.device, dtype=pred.dtype)
        pred = pred[mask].flatten(end_dim=-2)
        target = target[mask].flatten(end_dim=-2)
        loss = torch.nn.BCEWithLogitsLoss()
        return loss(pred, target)
