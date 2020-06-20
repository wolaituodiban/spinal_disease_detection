import torch


class KeyPointAcc:
    def __init__(self, max_dist=8, point=None):
        self.max_dist = max_dist
        self.point = point

    def __call__(self, pred, dist, mask):
        """

        :param pred: (batch_size, num_points, 2)
        :param dist: (batch_size, num_points, height, width)
        :param mask: (batch_size, num_points, 1, 1)
        :return:
        """
        width_indices = pred[:, :, 0].flatten()
        height_indices = pred[:, :, 1].flatten()
        image_indices = torch.arange(pred.shape[0], device=pred.device)
        image_indices = image_indices.unsqueeze(1).expand(-1, pred.shape[1]).flatten()
        point_indices = torch.arange(pred.shape[1], device=pred.device).repeat(pred.shape[0])

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
