import torch
from torch.nn.functional import interpolate
from torchvision.models.detection.backbone_utils import BackboneWithFPN


class PosModel(torch.nn.Module):
    def __init__(self, backbone: BackboneWithFPN, num_points: int):
        super().__init__()
        self.backbone = backbone
        self.fc = torch.nn.Conv2d(backbone.out_channels, num_points, kernel_size=1)

    def forward(self, images):
        images = images.to(self.fc.weight.device)
        feature_maps = self.backbone(images)
        output = self.fc(feature_maps['0'])
        output = interpolate(output, images.shape[-2:])
        if self.training:
            return output,
        else:
            return self._inference(output)

    @staticmethod
    def _inference(score):
        size = score.size()
        tensor = score.flatten(start_dim=2)
        max_indices = torch.argmax(tensor, dim=-1)
        indices0, indices1 = torch.where(max_indices > -1)
        indices2 = max_indices.flatten() // size[3]
        indices3 = max_indices.flatten() % size[3]
        return indices0, indices1, indices2, indices3
