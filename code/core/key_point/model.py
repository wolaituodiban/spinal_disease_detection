from typing import Dict, Tuple
import torch
from torch.nn.functional import interpolate
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .loss import KeyPointBCELoss
from .spinal_model import SpinalModel


class KeyPointModel(torch.nn.Module):
    def __init__(self, backbone: BackboneWithFPN, num_points: int, pixel_mean, pixel_std,
                 loss=KeyPointBCELoss(), spinal_model: SpinalModel = None, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Conv2d(backbone.out_channels, num_points, kernel_size=1)
        )
        self.register_buffer('pixel_mean', pixel_mean)
        self.register_buffer('pixel_std', pixel_std)
        self.spinal_model = spinal_model
        self.loss = loss

    @property
    def out_channels(self):
        return self.backbone.out_channels

    def set_spinal_model(self, spinal_model: SpinalModel):
        self.spinal_model = spinal_model

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.pixel_mean.device)
        images = (images - self.pixel_mean) / self.pixel_std
        images = images.expand(-1, 3, -1, -1)
        return images

    def cal_backbone(self, images: torch.Tensor) -> torch.Tensor:
        images = self._preprocess(images)
        output = self.backbone.body(images)
        return list(output.values())[-1]

    def cal_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        images = self._preprocess(images)
        feature_maps = self.backbone(images)
        return feature_maps['0']

    def cal_scores(self, images: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        feature_map = self.cal_feature_map(images)
        scores = self.fc(feature_map)
        scores = interpolate(scores, images.shape[-2:], mode='bilinear', align_corners=True)
        return scores, feature_map

    def forward(self, images, labels=None, masks=None) -> (torch.Tensor,):
        scores, _ = self.cal_scores(images)
        if self.training:
            return self.loss(scores, labels, masks),
        else:
            heatmaps = scores.sigmoid_()
            return self.spinal_model(heatmaps),


class UpSampleBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(channels, channels, 3, stride=2)
        self.bn = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU()

    def forward(self, images):
        output = self.conv(images)
        output = self.bn(output)
        output = self.relu(output)
        return output


class KeyPointModelV2(KeyPointModel):
    """
    没用
    """
    def forward(self, images, labels=None, masks=None):
        scores = self.cal_scores(images)
        if self.training:
            if self.loss is None:
                return scores
            else:
                return self.loss(scores, labels, masks),
        else:
            return self.spinal_model(scores),

    def cal_scores(self, images):
        images = images.to(self.pixel_mean.device)
        images = (images - self.pixel_mean) / self.pixel_std
        images = images.expand(-1, 3, -1, -1)
        feature_maps = self.backbone(images)
        scores = self.fc(feature_maps['0'])
        scores = interpolate(scores, images.shape[-2:], mode='bilinear', align_corners=True).sigmoid_()
        # 伪softmax
        background = 1 - scores.max(dim=1, keepdim=True)[0]
        scores = torch.cat([scores, background], dim=1)
        scores = torch.log(scores / (1 - scores))
        scores = scores.softmax(dim=1)[:, :-1]
        return scores
