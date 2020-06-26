import os
import torch
from typing import Dict, List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .dicom import DICOM
from .series import Series
from ..data_utils import read_annotation


class Study(dict):
    def __init__(self, study_dir):
        dicom_dict = {}
        for dicom_name in os.listdir(study_dir):
            dicom_path = os.path.join(study_dir, dicom_name)
            dicom = DICOM(dicom_path)
            series_uid = dicom.series_uid
            if series_uid not in dicom_dict:
                dicom_dict[series_uid] = [dicom]
            else:
                dicom_dict[series_uid].append(dicom)
        super().__init__({k: Series(v) for k, v in dicom_dict.items()})

        self.t2_sagittal_uid = None
        self.t2_transverse_uid = None
        # 通过平均值最大的来剔除压脂项
        max_t2_sagittal_mean = 0
        max_t2_transverse_mean = 0
        for series_uid, series in self.items():
            if series.plane == 'sagittal' and series.t_type == 'T2':
                t2_sagittal_mean = series.mean
                if t2_sagittal_mean > max_t2_sagittal_mean:
                    max_t2_sagittal_mean = t2_sagittal_mean
                    self.t2_sagittal_uid = series_uid
            if series.plane == 'transverse' and series.t_type == 'T2':
                t2_transverse_mean = series.mean
                if t2_transverse_mean > max_t2_transverse_mean:
                    max_t2_transverse_mean = t2_transverse_mean
                    self.t2_transverse_uid = series_uid

    @property
    def study_uid(self):
        return list(self.values())[0].study_uid

    @property
    def t2_sagittal(self) -> Series:
        return self[self.t2_sagittal_uid]

    @property
    def t2_transverse(self) -> Series:
        return self[self.t2_transverse_uid]

    @property
    def t2_sagittal_middle_frame(self) -> DICOM:
        return self.t2_sagittal.middle_frame

    def set_t2_sagittla_middle_frame(self, series_uid, instance_uid):
        self.t2_sagittal_uid = series_uid
        self.t2_sagittal.set_middle_frame(instance_uid)

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


def construct_studies(data_dir, annotation_path=None):
    """
    方便批量构造study的函数
    :param data_dir: 存放study的文件夹
    :param annotation_path: 如果有标注，那么根据标注来确定定位帧
    :return:
    """
    with Pool(cpu_count()) as pool:
        async_results = []
        for study_name in os.listdir(data_dir):
            study_dir = os.path.join(data_dir, study_name)
            async_results.append(pool.apply_async(Study, (study_dir, )))

        studies: Dict[str, Study] = {}
        for async_result in tqdm(async_results, ascii=True):
            async_result.wait()
            study = async_result.get()
            studies[study.study_uid] = study

    if annotation_path is None:
        return studies
    else:
        annotation = read_annotation(annotation_path)
        for k in annotation.keys():
            if k[0] in studies:
                studies[k[0]].set_t2_sagittla_middle_frame(k[1], k[2])
        return studies, annotation
