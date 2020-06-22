import math
import random
from typing import Any, Dict, Tuple
import torch
from PIL import Image
from ..data_utils import rotate_point


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


class SpinalModel:
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
        self.templates = []
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
            self.templates.append(key_point)
        self.templates = torch.stack(self.templates)
        self.num_candidates = num_candidates
        self.num_selected_templates = num_selected_templates
        self.max_translation = max_translation
        self.scale_range = scale_range
        self.max_angel = max_angel

    def __call__(self, heatmaps):
        size = heatmaps.size()
        flatten = heatmaps.flatten(start_dim=2)
        max_indices = torch.argmax(flatten, dim=-1)
        height_indices = max_indices.flatten() // size[3]
        width_indices = max_indices.flatten() % size[3]
        # 粗略估计
        preds = torch.stack([width_indices, height_indices], dim=1)
        preds = preds.reshape(flatten.shape[0], flatten.shape[1], 2)
        # 修正
        preds = [self.correct_prediction(preds[i], heatmaps[i]) for i in range(preds.shape[0])]
        preds = torch.stack(preds, dim=0)
        return preds

    def transform_templates(self, width, height, pred_points: torch.Tensor):
        """
        pred_points: (num_points, 2)
        return: (num_templates, num_points, 2)
        """
        pred_points = pred_points.to(self.templates.dtype)
        v_angel = cal_vertical_angle(pred_points[:, 0], pred_points[:, 1])
        templates = self.templates.clone().to(v_angel.device)
        # 伸缩
        templates[:, :, 0] *= width / self.STANDARD_SIZE
        templates[:, :, 1] *= height / self.STANDARD_SIZE
        # 旋转
        templates = rotate_point(templates, -v_angel, torch.zeros(2, device=v_angel.device))
        # 平移
        templates += pred_points.mean(dim=0, keepdim=True)
        return templates

    def generate_one_candidate(self, width, height, templates):
        """

        :param width: int
        :param height: int
        :param templates: (num_points, 2)
        :return:
        """
        selected_templates = templates[torch.randint(0, templates.shape[0], [self.num_selected_templates, ])]
        weights = torch.rand(self.num_selected_templates).to(templates.device)
        weights = torch.softmax(weights, dim=-1)
        weights = weights.reshape(-1, 1, 1)
        candidate = (weights * selected_templates).sum(dim=0)
        centor = candidate.mean(dim=0, keepdim=True)
        # 伸缩
        scale = random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        candidate *= scale
        # 旋转
        angel = random.randint(-self.max_angel, self.max_angel)
        candidate = rotate_point(candidate, angel, centor)
        # 平移
        centor[0, 0] += (random.random() - 0.5) * 2 * self.max_translation * width
        centor[0, 1] += (random.random() - 0.5) * 2 * self.max_translation * height
        candidate += centor - candidate.mean(dim=0, keepdim=True)
        return candidate

    def generate_candidates(self, width, height, pred_points: torch.Tensor):
        """

        :param width: int
        :param height: int
        :param pred_points: (num_points, 2)
        :return: (num_candidates, num_points, 2)
        """
        templates = self.transform_templates(width, height, pred_points)
        candidates = [self.generate_one_candidate(width, height, templates)
                      for _ in range(self.num_candidates)]
        candidates = torch.stack(candidates, dim=0).long()
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


