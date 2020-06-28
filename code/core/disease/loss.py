import torch


class DisLoss:
    def __call__(self, pred, target, mask):
        target = target.to(device=pred.device)
        pred = pred[mask]
        target = target[mask]
        loss = torch.nn.CrossEntropyLoss()
        return loss(pred, target)
