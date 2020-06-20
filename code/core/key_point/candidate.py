import math
import torch
from typing import Any, Dict, Tuple
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


class KeyPointTemplate:
    STANDARD_SIZE = 1000

    def __init__(self,
                 images: Dict[Any, Image.Image],
                 annotations: Dict[Any, Tuple[torch.Tensor, torch.Tensor]],
                 num_candidates=1000
                 ):
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
        self.point_indices = torch.arange(self.templates.shape[1])

    def transform_templates(self, width, height, pred_points: torch.Tensor):
        """
        pred_points: (11, 2)
        """
        v_angel = cal_vertical_angle(pred_points[:, 0], pred_points[:, 1])
        templates = self.templates.clone().to(v_angel.device)
        templates[:, :, 0] *= width / self.STANDARD_SIZE
        templates[:, :, 1] *= height / self.STANDARD_SIZE
        templates = rotate_point(templates, -v_angel, torch.zeros(2, device=v_angel.device))
        templates += pred_points.mean(dim=0, keepdim=True)
        return templates

    def generate_candidates(self, width, height, pred_points: torch.Tensor):
        templates = self.transform_templates(width, height, pred_points)
        return templates
