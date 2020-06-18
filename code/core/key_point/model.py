import torch
from torch.nn.functional import interpolate
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .loss import KeyPointBCELoss


class KeyPointModel(torch.nn.Module):
    def __init__(self, backbone: BackboneWithFPN, num_points: int, pixel_mean, pixel_std, loss=KeyPointBCELoss()):
        super().__init__()
        self.backbone = backbone
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(backbone.out_channels, num_points, kernel_size=1)
        )
        self.register_buffer('pixel_mean', pixel_mean)
        self.register_buffer('pixel_std', pixel_std)
        self.loss = loss

    def forward(self, images, labels=None, masks=None):
        scores = self.cal_scores(images)
        if self.training:
            return self.loss(scores, labels, masks),
        else:
            return self._inference(scores)

    def cal_scores(self, images):
        images = images.to(self.fc.weight.device)
        images = (images - self.pixel_mean) / self.pixel_std
        images = images.expand(-1, 3, -1, -1)
        feature_maps = self.backbone(images)
        scores = self.fc(feature_maps['0'])
        scores = interpolate(scores, images.shape[-2:])
        return scores

    @staticmethod
    def _inference(score):
        size = score.size()
        tensor = score.flatten(start_dim=2)
        max_indices = torch.argmax(tensor, dim=-1)
        indices0, indices1 = torch.where(max_indices > -1)
        indices2 = max_indices.flatten() // size[3]
        indices3 = max_indices.flatten() % size[3]
        return indices0, indices1, indices2, indices3
