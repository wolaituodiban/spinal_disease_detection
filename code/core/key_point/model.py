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
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(backbone.out_channels, num_points, kernel_size=1)
        )
        self.register_buffer('pixel_mean', pixel_mean)
        self.register_buffer('pixel_std', pixel_std)
        self.spinal_model = spinal_model
        self.loss = loss

    def set_spinal_model(self, spinal_model: SpinalModel):
        self.spinal_model = spinal_model

    def forward(self, images, labels=None, masks=None):
        scores = self.cal_scores(images)
        if self.training:
            return self.loss(scores, labels, masks),
        else:
            heatmaps = scores.sigmoid()
            return self.spinal_model(heatmaps),

    def cal_scores(self, images):
        images = images.to(self.pixel_mean.device)
        images = (images - self.pixel_mean) / self.pixel_std
        images = images.expand(-1, 3, -1, -1)
        feature_maps = self.backbone(images)
        scores = self.fc(feature_maps['0'])
        scores = interpolate(scores, images.shape[-2:], mode='bilinear', align_corners=True)
        return scores


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
    def __init__(self, backbone: BackboneWithFPN, num_points: int, pixel_mean, pixel_std,
                 loss, spinal_model: SpinalModel = None, dropout=0.1):
        super().__init__(backbone, num_points, pixel_mean, pixel_std, loss, spinal_model, dropout)
        channels = self.backbone.out_channels
        self.up_sample = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(channels, channels, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU()
        )

    def cal_scores(self, images):
        images = images.to(self.pixel_mean.device)
        images = (images - self.pixel_mean) / self.pixel_std
        images = images.expand(-1, 3, -1, -1)
        feature_maps = self.backbone(images)
        feature_maps = self.up_sample(feature_maps['0'])
        scores = self.fc(feature_maps)
        scores = interpolate(scores, images.shape[-2:], mode='bilinear', align_corners=True)
        return scores
