class KeyPointAcc:
    def __init__(self, max_dist=8, point=None):
        self.max_dist = max_dist
        self.point = point

    def __call__(self, image_indices, point_indices, height_indices, width_indices, dist, mask):
        """

        :param image_indices:
        :param point_indices:
        :param height_indices:
        :param width_indices:
        :param dist: (batch_size, num_points, height, width)
        :param mask: (batch_size, num_points, 1, 1)
        :return:
        """
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

