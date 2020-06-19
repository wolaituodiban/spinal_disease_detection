class KeyPointAcc:
    def __init__(self, max_dist=8, point=None):
        self.max_dist = max_dist
        self.point = point

    def __call__(self, indices0, indices1, indices2, indices3, dist, mask):
        """

        :param indices0:
        :param indices1:
        :param indices2:
        :param indices3:
        :param dist: (batch_size, num_points, height, width)
        :param mask: (batch_size, num_points, 1, 1)
        :return:
        """
        if self.point is not None:
            point_mask = indices1 == self.point
            indices0 = indices0[point_mask]
            indices1 = indices1[point_mask]
            indices2 = indices2[point_mask]
            indices3 = indices3[point_mask]

        dist = dist.to(indices0.device)
        mask = mask.to(indices0.device)

        mask = mask[indices0, indices1]
        dist = dist[indices0, indices1, indices2, indices3]
        dist = dist[mask.flatten()]
        return (dist < self.max_dist).float().mean()

