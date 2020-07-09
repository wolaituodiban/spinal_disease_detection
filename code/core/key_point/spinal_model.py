import math
from typing import Any, Dict, Tuple
import torch
from PIL import Image
from ..data_utils import rotate_point, rotate_batch


def cal_vertical_angle(x, y):
    """
    计算一系列点的拟合直线与垂直线之间的夹角
    """
    x = x - x.mean()
    y = y - y.mean()
    xx = (x ** 2).sum()
    xy = (x * y).sum()
    yy = (y ** 2).sum()
    lamb = ((xx + yy) - ((xx - yy) ** 2 + 4 * xy * xy).sqrt()) / 2
    return torch.atan((lamb - xx) / xy) / math.pi * 180


class SpinalModelBase(torch.nn.Module):
    @staticmethod
    def forward(heatmaps: torch.Tensor):
        """

        :param heatmaps: (num_batch, num_points, height, width)
        :return: (num_batch, num_points, 2)
        """
        size = heatmaps.size()
        flatten = heatmaps.flatten(start_dim=2)
        max_indices = torch.argmax(flatten, dim=-1)
        height_indices = max_indices.flatten() // size[3]
        width_indices = max_indices.flatten() % size[3]
        # 粗略估计
        preds = torch.stack([width_indices, height_indices], dim=1)
        preds = preds.reshape(flatten.shape[0], flatten.shape[1], 2)
        return preds


class SpinalModel(SpinalModelBase):
    """
    通过对脊柱模板的伸缩,平移,旋转和线性组合,生成候选脊柱坐标
    """
    STANDARD_SIZE = 1000

    def __init__(self,
                 images: Dict[Any, Image.Image],
                 annotations: Dict[Any, Tuple[torch.Tensor, torch.Tensor]],
                 num_candidates=128,
                 num_selected_templates=8,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 max_angel=10,
                 max_translation: float = 0.05):
        super().__init__()
        templates = []
        for k, annotation in annotations.items():
            width, height = images[k].size
            key_point = torch.cat([coord[:, :2] for coord in annotation], dim=0)
            key_point = key_point.to(torch.float32)
            key_point *= 2 * self.STANDARD_SIZE / (width + height)
            # 将中心平移到原点
            key_point -= key_point.mean(dim=0, keepdim=True)
            # 计算拟合直线与纵坐标的夹角
            v_angel = cal_vertical_angle(key_point[:, 0], key_point[:, 1])
            # 旋转坐标，使得拟合直线与纵坐标重合
            key_point = rotate_point(key_point, v_angel, torch.zeros(2))
            templates.append(key_point)
        self.register_buffer('templates', torch.stack(templates))
        self.num_candidates = num_candidates
        self.num_selected_templates = num_selected_templates
        self.max_translation = max_translation
        self.scale_range = scale_range
        self.max_angel = max_angel

    def forward(self, heatmaps):
        preds = super().forward(heatmaps)
        # 修正
        preds = preds.to(device=self.templates.device, dtype=torch.float32)
        preds = [self.correct_prediction(preds[i], heatmaps[i]) for i in range(preds.shape[0])]
        preds = torch.stack(preds, dim=0)
        return preds.to(device=heatmaps.device)

    def transform_templates(self, width, height, pred_points: torch.Tensor) -> torch.Tensor:
        """
        pred_points: (num_points, 2)
        return: (num_templates, num_points, 2)
        """
        # 伸缩
        ratio = torch.tensor([width / self.STANDARD_SIZE, height / self.STANDARD_SIZE], device=self.templates.device)
        templates = self.templates * ratio

        # 旋转
        v_angel = cal_vertical_angle(pred_points[:, 0], pred_points[:, 1])
        templates = rotate_point(templates, -v_angel, torch.zeros(2, device=v_angel.device))
        # 平移
        templates += pred_points.mean(dim=0, keepdim=True)
        return templates

    def linear_combination(self, templates: torch.Tensor) -> torch.Tensor:
        """
        
        :param templates: (num_templates, num_points, 2) 
        :return: (num_candidates, num_points, 2)
        """
        indices = torch.randint(0, templates.shape[0], [self.num_candidates, self.num_selected_templates])
        selected_templates = templates[indices]
        weights = torch.rand(indices.shape, device=templates.device)
        weights = torch.softmax(weights, dim=-1)
        weights = weights.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return (weights * selected_templates).sum(dim=1)
    
    def transform_candidates(self, width, height, candidates: torch.Tensor) -> torch.Tensor:
        """
        
        :param width: 
        :param height: 
        :param candidates: (num_candidates, num_points, 2)
        :return: (num_candidates, num_selected_templates, num_points, 2) 
        """
        centers = candidates.mean(dim=1)
        angels = torch.randint(-self.max_angel, self.max_angel, candidates.shape[:1], device=candidates.device)
        # 伸缩
        scales = torch.rand(candidates.shape[0], device=candidates.device)
        scales = scales * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        scales = scales.unsqueeze(dim=-1).unsqueeze(dim=-1)
        output = candidates * scales
        # 旋转

        output = rotate_batch(output, angels, centers)
        # 平移
        shift = (torch.rand(centers.shape, device=candidates.device) - 0.5) * 2 * self.max_translation
        shift *= torch.tensor([[width, height]], device=candidates.device)
        centers += shift
        output += centers.unsqueeze(1) - output.mean(dim=1, keepdim=True)
        return output

    def generate_candidates(self, width, height, pred_points: torch.Tensor):
        """

        :param width: int
        :param height: int
        :param pred_points: (num_points, 2)
        :return: (num_candidates, num_points, 2)
        """
        templates = self.transform_templates(width, height, pred_points)
        candidates = self.linear_combination(templates)
        candidates = self.transform_candidates(width, height, candidates)
        candidates = candidates.long()
        # 防止越界
        candidates[:, :, 0] = torch.clamp(candidates[:, :, 0], 0, width - 1)
        candidates[:, :, 1] = torch.clamp(candidates[:, :, 1], 0, height - 1)
        return candidates

    @staticmethod
    def score_candidates(candidates, heatmap):
        """
        candidates: (num_candidates, num_points, 2)
        heatmap: (num_points, height, width)
        return: (num_candidates, )
        """
        point_indices = torch.arange(candidates.shape[1], device=candidates.device)
        point_indices = point_indices.repeat(candidates.shape[0])
        width_indices = candidates[:, :, 0].flatten()
        height_indices = candidates[:, :, 1].flatten()
        assert width_indices.max() < heatmap.shape[2] and height_indices.max() < heatmap.shape[1]
        scores = heatmap[point_indices, height_indices, width_indices]
        scores = scores.reshape(candidates.shape[0], -1)
        scores = torch.cumprod(scores, dim=-1)[:, -1]
        return scores

    def correct_prediction(self, pred_points: torch.Tensor, heatmap: torch.Tensor):
        """
        pred_points: (num_points, 2)
        heatmap: (num_points, height, width)
        return: (num_points, 2)
        """
        assert len(pred_points.shape) == 2 and pred_points.shape[1] == 2
        assert len(heatmap.shape) == 3
        height, width = heatmap.shape[-2:]
        candidates = self.generate_candidates(width, height, pred_points)
        scores = self.score_candidates(candidates, heatmap)

        max_idx = torch.argmax(scores)
        return candidates[max_idx]
