import torch
from tqdm import tqdm


class KeyPointAcc:
    def __init__(self, max_dist=6, point=None):
        """

        :param max_dist: 判定为真的距离阈值
        :param point: 如果不是none，那么之计算某个点的精度
        """
        self.max_dist = max_dist
        self.point = point

    def __call__(self, vertebra_coords, disc_coords, dist, mask):
        """

        :param vertebra_coords: (batch_size, num_points, 2)
        :parms disc_coords: (batch_size, num_points, 2)
        :param dist: (batch_size, num_points, height, width)
        :param mask: (batch_size, num_points, 1, 1)
        :return:
        """
        pred_coords = torch.cat([vertebra_coords, disc_coords], dim=1)
        width_indices = pred_coords[:, :, 0].flatten()
        height_indices = pred_coords[:, :, 1].flatten()
        image_indices = torch.arange(pred_coords.shape[0], device=pred_coords.device)
        image_indices = image_indices.unsqueeze(1).expand(-1, pred_coords.shape[1]).flatten()
        point_indices = torch.arange(pred_coords.shape[1], device=pred_coords.device).repeat(pred_coords.shape[0])

        if self.point is not None:
            point_mask = point_indices == self.point
            image_indices = image_indices[point_mask]
            point_indices = point_indices[point_mask]
            height_indices = height_indices[point_mask]
            width_indices = width_indices[point_mask]

        dist = dist.to(image_indices.device)
        mask = mask.to(image_indices.device)

        mask = mask[image_indices, point_indices]
        dist = dist[image_indices, point_indices, height_indices, width_indices]
        dist = dist[mask.flatten()]
        return (dist < self.max_dist).float().mean()


def distance_distribution(module, data_loader):
    with torch.no_grad():
        dists = []
        module.eval()
        for data, (dist, mask) in tqdm(data_loader, ascii=True):
            vertebra_coords, disc_coords = module(*data)
            pred = torch.cat([vertebra_coords, disc_coords], dim=1)
            width_indices = pred[:, :, 0].flatten()
            height_indices = pred[:, :, 1].flatten()
            image_indices = torch.arange(pred.shape[0], device=pred.device)
            image_indices = image_indices.unsqueeze(1).expand(-1, pred.shape[1]).flatten()
            point_indices = torch.arange(pred.shape[1], device=pred.device).repeat(pred.shape[0])

            dist = dist.to(image_indices.device)
            mask = mask.to(image_indices.device)

            mask = mask[image_indices, point_indices]
            dist = dist[image_indices, point_indices, height_indices, width_indices]
            dist = dist[mask.flatten()]
            dists.append(dist)
        dists = torch.cat(dists).flatten()
    return dists
