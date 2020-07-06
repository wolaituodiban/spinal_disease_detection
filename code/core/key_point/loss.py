import math
import torch


class NullLoss:
    def __call__(self, x, *args):
        return x.mean()


class KeyPointBCELoss:
    def __init__(self, max_dist=8):
        self.max_dist = max_dist

    def __call__(self, pred: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor):
        dist = dist.to(pred.device)

        pred = pred[mask]
        dist = dist[mask]
        label = dist < self.max_dist
        label = label.to(pred.dtype)

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=1 / label.mean())
        return loss(pred, label)


class KeyPointBCELossV2:
    def __init__(self, lamb=8/math.log(2)):
        self.lamb = lamb

    def __call__(self, pred: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor):
        dist = dist.to(pred.device)

        pred = pred[mask]
        dist = dist[mask]
        label = 1 / (dist / self.lamb).exp()

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=1/label.mean())
        return loss(pred, label)


class CascadeLoss:
    def __init__(self):
        self.loss = torch.nn.SmoothL1Loss()

    def __call__(self, pred_coords, gt_coords, masks, size):
        """
        坐标都是先宽后高，但是这里的size是先高后宽
        :param pred_coords: (batch_size, num_points, 2)
        :param gt_coords: 与pred_coords相同
        :param masks: 与pred_coords相同
        :param size: 图片的大小(height, width)
        :return:
        """
        size = size[[1, 0]]
        return self.loss(pred_coords[masks] / size, gt_coords[masks] / size)


class CascadeLossV2(CascadeLoss):
    """
    坐标差分损失，对相邻两个点坐标的差进行损失计算，实际上就是对向量的回归

    由于最初开发时的不良设计，如果将锥体和椎间盘从上到下依次编号0到10，那么
    pred_coords每一行对应的编号是1，3，5，7，9，0，2，4，6，8，10。如果想要将pred_coords的的顺序
    恢复原状，那么需要将行按照如下顺序重新排列：5，0，6，1，7，2，8，3，9，4，10

    我将这个序列保存在类当中，并在__call__函数中，先按照这个顺序对pred_coords进行重排列，
    然后再进行差分
    """
    def __init__(self, degree: int = 1):
        super(CascadeLossV2, self).__init__()
        self.degree = degree
        self.__reindex__ = [5, 0, 6, 1, 7, 2, 8, 3, 9, 4, 10]

    def __call__(self, pred_coords, gt_coords, masks, size):
        pred_coords = pred_coords[:, self.__reindex__]
        gt_coords = gt_coords[:, self.__reindex__]
        masks = masks[:, self.__reindex__]

        all_pred = [pred_coords]
        all_gt = [gt_coords]
        all_masks = [masks]
        for degree in range(1, self.degree+1):
            pred_diff = pred_coords[:, degree:] - pred_coords[:, :-degree]
            gt_diff = gt_coords[:, degree:] - gt_coords[:, :-degree]
            masks_diff = torch.bitwise_and(masks[:, degree:], masks[:, :-degree])

            all_pred.append(pred_diff)
            all_gt.append(gt_diff)
            all_masks.append(masks_diff)

        all_pred = torch.cat(all_pred, dim=1)
        all_gt = torch.cat(all_gt, dim=1)
        all_masks = torch.cat(all_masks, dim=1)
        return super(CascadeLossV2, self).__call__(all_pred, all_gt, all_masks, size)
