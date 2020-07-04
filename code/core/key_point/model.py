from typing import List
import torch
from torch.nn.functional import interpolate
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .loss import KeyPointBCELoss, CascadeLoss
from .spinal_model import SpinalModelBase
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_DISC_ID


def extract_point_feature(feature_maps: torch.Tensor, coords, height, width):
    """
    :param feature_maps: (batch_size, channels, height, width)
    :param coords: (batch_size, n_points, 2), width在前，height在后
    :param height:
    :param width:
    :return: (batch_size, n_points, channels)
    """
    ratio = torch.tensor([feature_maps.shape[-2] / height, feature_maps.shape[-1] / width], device=coords.device)
    # 需要调整width, height的顺序
    coords = (coords[:, :, [1, 0]] * ratio).round().long()
    image_indices = torch.arange(coords.shape[0]).unsqueeze(1).expand(-1, coords.shape[1]).flatten()
    width_indices = coords[:, :, 1].flatten()
    height_indices = coords[:, :, 0].flatten()
    features = feature_maps.permute(0, 2, 3, 1)
    features = features[image_indices, height_indices, width_indices]
    features = features.reshape(*coords.shape[:2], -1)
    return features


class KeyPointModel(torch.nn.Module):
    def __init__(self, backbone: BackboneWithFPN, num_vertebra_points: int = len(SPINAL_VERTEBRA_ID),
                 num_disc_points: int = len(SPINAL_DISC_ID), pixel_mean=0.5, pixel_std=1,
                 loss=KeyPointBCELoss(), spinal_model=SpinalModelBase()):
        super().__init__()
        self.backbone = backbone
        self.num_vertebra_points = num_vertebra_points
        self.num_disc_point = num_disc_points
        self.fc = torch.nn.Conv2d(backbone.out_channels, num_vertebra_points + num_disc_points, kernel_size=1)
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean))
        self.register_buffer('pixel_std', torch.tensor(pixel_std))
        self.spinal_model = spinal_model
        self.loss = loss

    @property
    def out_channels(self):
        return self.backbone.out_channels

    def kp_parameters(self):
        for p in self.fc.parameters():
            yield p

    def set_spinal_model(self, spinal_model: SpinalModelBase):
        self.spinal_model = spinal_model

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.pixel_mean.device)
        images = (images - self.pixel_mean) / self.pixel_std
        images = images.expand(-1, 3, -1, -1)
        return images

    def cal_scores(self, images):
        images = self._preprocess(images)
        feature_pyramids = self.backbone(images)
        feature_maps = feature_pyramids['0']
        scores = self.fc(feature_maps)
        scores = interpolate(scores, images.shape[-2:], mode='bilinear', align_corners=True)
        return scores, feature_maps

    def cal_backbone(self, images: torch.Tensor) -> torch.Tensor:
        images = self._preprocess(images)
        output = self.backbone.body(images)
        return list(output.values())[-1]

    def pred_coords(self, scores, split=True):
        heat_maps = scores.sigmoid()
        coords = self.spinal_model(heat_maps)
        if split:
            vertebra_coords = coords[:, :self.num_vertebra_points]
            disc_coords = coords[:, self.num_vertebra_points:]
            return vertebra_coords, disc_coords, heat_maps
        else:
            return coords, heat_maps

    def forward(self, images, distmaps=None, masks=None, return_more=False) -> tuple:
        scores, feature_maps = self.cal_scores(images)
        if self.training:
            if distmaps is None:
                loss = None
            else:
                loss = self.loss(scores, distmaps, masks)
            if return_more:
                vertebra_coords, disc_coords, heat_maps = self.pred_coords(scores)
                return loss, vertebra_coords, disc_coords, heat_maps, feature_maps
            else:
                return loss,
        else:
            vertebra_coords, disc_coords, heat_maps = self.pred_coords(scores)
            if return_more:
                return vertebra_coords, disc_coords, heat_maps, feature_maps
            else:
                return vertebra_coords, disc_coords


class KeyPointModelV2(KeyPointModel):
    def __init__(self, backbone: BackboneWithFPN, num_vertebra_points: int = len(SPINAL_VERTEBRA_ID),
                 num_disc_points: int = len(SPINAL_DISC_ID), pixel_mean=0.5, pixel_std=1,
                 loss=KeyPointBCELoss(), spinal_model=SpinalModelBase(), num_cascades=1,
                 cascade_loss=CascadeLoss(), loss_scaler=1):
        super().__init__(backbone, num_vertebra_points, num_disc_points, pixel_mean, pixel_std, loss, spinal_model)
        cascade_heads = []
        for _ in range(num_cascades):
            head = torch.nn.Linear(backbone.out_channels, 2)
            cascade_heads.append(head)
        self.cascade_heads = torch.nn.ModuleList(cascade_heads)
        self.cascade_loss = cascade_loss
        self.loss_scaler = loss_scaler
        self.spinal_model_base = SpinalModelBase()

    def kp_parameters(self):
        for p in super().kp_parameters():
            yield p
        for p in self.cascade_heads.parameters():
            yield p

    def run_cascades(self, feature_maps, coords, height, width) -> List[torch.Tensor]:
        outputs = []
        for head in self.cascade_heads:
            points_features = extract_point_feature(feature_maps, coords, height, width)
            shift = head(points_features)
            coords = coords + shift
            outputs.append(coords)
        return outputs

    def collect_cascades_losses(self, cascades_outputs: List[torch.Tensor], distmaps, masks) -> List[torch.Tensor]:
        losses = []
        gt_coords = self.spinal_model_base(-distmaps).float()
        size = torch.tensor(distmaps.shape[-2:], dtype=torch.float32, device=distmaps.device)
        for pred_coords in cascades_outputs:
            loss = self.cascade_loss(pred_coords, gt_coords, masks, size) * self.loss_scaler
            losses.append(loss)
        return losses

    def forward(self, images, distmaps=None, masks=None, return_more=False) -> tuple:
        scores, feature_maps = self.cal_scores(images)
        coords, heat_maps = self.pred_coords(scores, split=False)
        cascades_outputs = self.run_cascades(feature_maps, coords, *images.shape[-2:])
        if self.training:
            if distmaps is None:
                loss = None
            else:
                distmaps = distmaps.to(scores.device)
                loss = self.loss(scores, distmaps, masks)
                cascades_losses = self.collect_cascades_losses(cascades_outputs, distmaps, masks)
                loss = torch.stack([loss] + cascades_losses, dim=0)
            if return_more:
                final_coords = cascades_outputs[-1].round().long()
                vertebra_coords = final_coords[:, :self.num_vertebra_points]
                disc_coords = final_coords[:, self.num_vertebra_points:]
                return loss, vertebra_coords, disc_coords, heat_maps, feature_maps
            else:
                return loss,
        else:
            final_coords = cascades_outputs[-1].round().long()
            vertebra_coords = final_coords[:, :self.num_vertebra_points]
            disc_coords = final_coords[:, self.num_vertebra_points:]
            if return_more:
                return vertebra_coords, disc_coords, heat_maps, feature_maps
            else:
                return vertebra_coords, disc_coords
