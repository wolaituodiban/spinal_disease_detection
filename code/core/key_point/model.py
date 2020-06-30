import torch
from torch.nn.functional import interpolate
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .loss import KeyPointBCELoss
from .spinal_model import SpinalModelBase
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_DISC_ID


class KeyPointModel(torch.nn.Module):
    def __init__(self, backbone: BackboneWithFPN, num_vertebra_points: int = len(SPINAL_VERTEBRA_ID),
                 num_disc_points: int = len(SPINAL_DISC_ID), pixel_mean=0.5, pixel_std=1,
                 loss=KeyPointBCELoss(), spinal_model=SpinalModelBase(), dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.num_vertebra_points = num_vertebra_points
        self.num_disc_point = num_disc_points
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Conv2d(backbone.out_channels, num_vertebra_points + num_disc_points, kernel_size=1)
        )
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean))
        self.register_buffer('pixel_std', torch.tensor(pixel_std))
        self.spinal_model = spinal_model
        self.loss = loss

    @property
    def out_channels(self):
        return self.backbone.out_channels

    def set_spinal_model(self, spinal_model: SpinalModelBase):
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

    def pred_coords(self, scores) -> (torch.Tensor, torch.Tensor):
        heatmaps = scores.sigmoid_()
        coords = self.spinal_model(heatmaps)
        vertebra_coords = coords[:, :self.num_vertebra_points]
        disc_coords = coords[:, self.num_vertebra_points:]
        return vertebra_coords, disc_coords

    def forward(self, images, distmaps=None, masks=None, return_more=False):
        images = self._preprocess(images)
        feature_pyramids = self.backbone(images)
        feature_maps = feature_pyramids['0']
        scores = self.fc(feature_maps)
        scores = interpolate(scores, images.shape[-2:], mode='bilinear', align_corners=True)
        if self.training:
            loss = self.loss(scores, distmaps, masks)
            if return_more:
                vertebra_coords, disc_coords = self.pred_coords(scores)
                return loss, feature_maps, vertebra_coords, disc_coords
            else:
                return loss,
        else:
            return self.pred_coords(scores)
