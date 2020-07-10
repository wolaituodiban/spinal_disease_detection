import torch


class DisLoss:
    def __init__(self, weight: list = None):
        if weight is not None:
            weight = torch.tensor(weight)
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(self, pred, target, mask):
        self.loss.to(pred.device)
        target = target.to(device=pred.device)
        pred = pred[mask]
        target = target[mask]
        return self.loss(pred, target)
