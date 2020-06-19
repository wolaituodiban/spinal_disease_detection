class KeyPointHitRate:
    def __init__(self, max_dist=8):
        self.max_dist = max_dist

    def __call__(self, indices0, indices1, indices2, indices3, dist, mask):
        dist = dist.to(indices0.device)
        mask = mask.to(indices0.device)

        mask = mask[indices0, indices1]
        dist = dist[indices0, indices1, indices2, indices3]
        dist = dist[mask.flatten()]
        return (dist < self.max_dist).float().mean()
