import torch


class DisLoss:
    def __init__(self, weight):
        self.weight = torch.tensor(weight)

    def __call__(self, pred, target, mask):
        loss = torch.nn.CrossEntropyLoss(weight=self.weight.to(pred.device))
        target = target.to(device=pred.device)
        pred = pred[mask]
        target = target[mask]
        return loss(pred, target)
