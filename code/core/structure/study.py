import os
import torch
from typing import List
from .dicom import DICOM
from .series import Series


class Study(dict):
    def __init__(self, study_dir):
        dicom_dict = {}
        for dicom_name in os.listdir(study_dir):
            dicom_path = os.path.join(study_dir, dicom_name)
            try:
                dicom = DICOM(dicom_path)
            except RuntimeError as e:
                print(dicom_path)
                print(e)
                continue
            series_uid = dicom.series_uid
            if series_uid not in dicom_dict:
                dicom_dict[series_uid] = [dicom]
            else:
                dicom_dict[series_uid].append(dicom)
        super().__init__({k: Series(v) for k, v in dicom_dict.items()})

        self.t2_sagittal_uid = None
        self.t2_transverse_uid = None
        # 可能存在很多t2_transverse，由于目标图像信号最强，所以选平均值最大的
        max_t2_transverse_mean = 0
        for series_uid, series in dicom_dict.items():
            series = Series(series)
            if series.plane == 'sagittal' and series.t_type == 'T2':
                self.t2_sagittal_uid = series_uid
            if series.plane == 'transverse' and series.t_type == 'T2':
                t2_transverse_mean = series.mean
                if t2_transverse_mean > max_t2_transverse_mean:
                    max_t2_transverse_mean = t2_transverse_mean
                    self.t2_transverse_uid = series_uid

    @property
    def study_uid(self):
        return self.t2_sagittal[0].study_uid

    @property
    def t2_sagittal(self) -> Series:
        return self[self.t2_sagittal_uid]

    @property
    def t2_transverse(self) -> Series:
        return self[self.t2_transverse_uid]

    @property
    def t2_sagittal_middle_frame(self) -> DICOM:
        return self.t2_sagittal.middle_frame

    def nearest_t2_sagittal(self, coord: torch.Tensor, k: int) -> List[List[DICOM]]:
        """
        给定几个在t2_sagittal_middle_frame预测像素坐标，返回对应的最近的几个t2_sagittal
        :param coord: 像素坐标，Nx2的矩阵或者长度为2的向量
        :param k: 每个坐标返回k个frame
        :return:
        """
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(coord)
        distance = self.t2_transverse.point_distance(human_coord)
        indices = torch.argsort(distance, dim=-1)
        if len(indices.shape) == 1:
            return [[self.t2_transverse[idx] for idx in indices[:k]]]
        else:
            return [[self.t2_transverse[idx] for idx in row[:k]] for row in indices]
