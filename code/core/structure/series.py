import torch
from collections import Counter
from typing import List
from .dicom import DICOM


class Series(list):
    """
    将DICOM的序列，并且会按照dim的方向，根据image_position对DICOM进行排列
    """
    def __init__(self, dicom_list: List[DICOM]):
        planes = [dicom.plane for dicom in dicom_list]
        plane_counter = Counter(planes)
        self.plane = plane_counter.most_common(1)[0][0]

        if self.plane == 'transverse':
            dim = 2
        elif self.plane == 'sagittal':
            dim = 0
        elif self.plane == 'transverse':
            dim = 1
        else:
            dim = None

        if dim is not None:
            dicom_list = sorted(dicom_list, key=lambda x: x.image_position[dim])
        super().__init__(dicom_list)
        self.instance_uids = {d.instance_uid: i for i, d in enumerate(self)}
        self.unit_normal_vectors = torch.stack([d.unit_normal_vector for d in self], dim=0)
        self.image_positions = torch.stack([d.image_position for d in self], dim=0)

        self.middle_frame_uid = None

    def __getitem__(self, item) -> DICOM:
        if isinstance(item, str):
            item = self.instance_uids[item]
        return super().__getitem__(item)

    @property
    def t_type(self):
        t_type_counter = Counter([d.t_type for d in self])
        return t_type_counter.most_common(1)[0][0]

    @property
    def mean(self):
        output = 0
        i = 0
        for dicom in self:
            temp = dicom.mean
            if temp is None:
                continue
            output += i / (i + 1) * output + temp / (i + 1)
            i += 1
        return output

    @property
    def middle_frame(self) -> DICOM:
        if self.middle_frame_uid is not None:
            return self[self.middle_frame_uid]
        else:
            return self[(len(self) - 1) // 2]

    def set_middle_frame(self, instance_uid):
        self.middle_frame_uid = instance_uid

    @property
    def study_uid(self):
        study_uid_counter = Counter([d.study_uid for d in self])
        return study_uid_counter.most_common(1)[0][0]

    @property
    def series_uid(self):
        study_uid_counter = Counter([d.study_uid for d in self])
        return study_uid_counter.most_common(1)[0][0]

    def point_distance(self, coord: torch.Tensor):
        """
        点到序列中每一张图像平面的距离，单位为毫米
        :param coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return: 长度为NxM的矩阵或者长度为M的向量，M是序列的长度
        """
        return ((coord.unsqueeze(-2) - self.image_positions) * self.unit_normal_vectors).sum(-1).abs()

    def k_nearest(self, coord: torch.Tensor, k) -> List[List[DICOM]]:
        """

        :param coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :param k:
        :return:
        """
        distance = self.point_distance(coord)
        indices = torch.argsort(distance, dim=-1)
        if len(indices.shape) == 1:
            return [[self[i] for i in indices[:k]]]
        else:
            return [[self[i] for i in row[:k]] for row in indices]

