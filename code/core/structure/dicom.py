import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from ..dicom_utils import dicom_metainfo_v3, dicom2array


def str2tensor(s: str) -> torch.Tensor:
    """

    :param s: numbers separated by '\\', eg.  '0.71875\\0.71875 '
    :return: 1-D tensor
    """
    return torch.tensor(list(map(float, s.split('\\'))))


def unit_vector(tensor: torch.Tensor, dim=-1):
    norm = (tensor ** 2).sum(dim=dim, keepdim=True).sqrt()
    return tensor / norm


def unit_normal_vector(orientation: torch.Tensor):
    temp1 = orientation[:, [1, 2, 0]]
    temp2 = orientation[:, [2, 0, 1]]
    output = temp1 * temp2[[1, 0]]
    output = output[0] - output[1]
    return unit_vector(output, dim=-1)


class DICOM:
    """
    解析dicom文件
    属性：
        study_uid：检查ID
        series_uid：序列ID
        instance_uid：图像ID
        series_description：序列描述，用于区分T1、T2等
        pixel_spacing: 长度为2的向量，像素的物理距离，单位是毫米
        image_position：长度为3的向量，图像左上角在人坐标系上的坐标，单位是毫米
        image_orientation：2x3的矩阵，第一行表示图像从左到右的方向，第二行表示图像从上到下的方向，单位是毫米？
        unit_normal_vector: 长度为3的向量，图像的单位法向量，单位是毫米？
        image：PIL.Image.Image，图像
    注：人坐标系，规定人体的左边是X轴的方向，从面部指向背部的方向表示y轴的方向，从脚指向头的方向表示z轴的方向
    """

    def __init__(self, file_path):
        self.file_path = file_path
        metainfo, msg = dicom_metainfo_v3(file_path)
        self.error_msg = msg
        self.study_uid: str = metainfo['studyUid'] or ''
        self.series_uid: str = metainfo['seriesUid'] or ''
        self.instance_uid: str = metainfo['instanceUid'] or ''
        self.series_description: str = metainfo['seriesDescription'] or ''

        if metainfo['pixelSpacing'] is None:
            self.pixel_spacing = torch.full([2, ], fill_value=np.nan)
        else:
            self.pixel_spacing = str2tensor(metainfo['pixelSpacing'])

        if metainfo['imagePosition'] is None:
            self.image_position = torch.full([3, ], fill_value=np.nan)
        else:
            self.image_position = str2tensor(metainfo['imagePosition'])

        if metainfo['imageOrientation'] is None:
            self.image_orientation = torch.full([2, 3], fill_value=np.nan)
            self.unit_normal_vector = torch.full([3, ], fill_value=np.nan)
        else:
            self.image_orientation = unit_vector(
                str2tensor(metainfo['imageOrientation']).reshape(2, 3))
            self.unit_normal_vector = unit_normal_vector(self.image_orientation)

        try:
            self.image: Image.Image = tf.to_pil_image(dicom2array(file_path))
        except RuntimeError as e:
            self.error_msg += str(e)
            self.image = None

    @property
    def t_type(self):
        if 'T1' in self.series_description.upper():
            return 'T1'
        elif 'T2' in self.series_description.upper():
            return 'T2'
        else:
            return None

    @property
    def plane(self):
        if torch.isnan(self.unit_normal_vector).all():
            return None
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 0., 1.])).abs() > 0.75:
            # 轴状位，水平切开
            return 'transverse'
        elif torch.matmul(self.unit_normal_vector, torch.tensor([1., 0., 0.])).abs() > 0.75:
            # 矢状位，左右切开
            return 'sagittal'
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 1., 0.])).abs() > 0.75:
            # 冠状位，前后切开
            return 'coronal'
        else:
            # 不知道
            return None

    @property
    def mean(self):
        if self.image is None:
            return None
        else:
            return tf.to_tensor(self.image).mean()

    @property
    def size(self):
        """

        :return: width and height
        """
        if self.image is None:
            return None
        else:
            return self.image.size

    def pixel_coord2human_coord(self, coord: torch.Tensor) -> torch.Tensor:
        """
        将图像上的像素坐标转换成人坐标系上的坐标
        :param coord: 像素坐标，Nx2的矩阵或者长度为2的向量
        :return: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        """
        return torch.matmul(coord * self.pixel_spacing, self.image_orientation) + self.image_position

    def point_distance(self, coord: torch.Tensor) -> torch.Tensor:
        """
        点到图像平面的距离，单位为毫米
        :param coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return: 长度为N的向量或者标量
        """
        return torch.matmul(coord - self.image_position, self.unit_normal_vector).abs()

    def projection(self, coord: torch.Tensor) -> torch.Tensor:
        """
        将人坐标系中的点投影到图像上，并输出像素坐标
        :param coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return:像素坐标，Nx2的矩阵或者长度为2的向量
        """
        cos = torch.matmul(coord - self.image_position, self.image_orientation.transpose(0, 1))
        return (cos / self.pixel_spacing).round()
