import os
import random
from multiprocessing import Pool, cpu_count
from typing import Union
from tqdm import tqdm
import torch
from torchvision.transforms import functional as tf
from .dicom import DICOM
from .series import Series
from ..data_utils import read_annotation


class Study(dict):
    def __init__(self, study_dir):
        with Pool(cpu_count()) as pool:
            async_results = []
            for dicom_name in os.listdir(study_dir):
                dicom_path = os.path.join(study_dir, dicom_name)
                async_results.append(pool.apply_async(DICOM, (dicom_path, )))

            dicom_dict = {}
            for async_result in async_results:
                async_result.wait()
                dicom = async_result.get()
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

        if self.t2_sagittal_uid is None:
            for series_uid, series in self.items():
                if series.plane == 'sagittal':
                    t2_sagittal_mean = series.mean
                    if t2_sagittal_mean > max_t2_sagittal_mean:
                        max_t2_sagittal_mean = t2_sagittal_mean
                        self.t2_sagittal_uid = series_uid

        if self.t2_transverse_uid is None:
            for series_uid, series in self.items():
                if series.plane == 'transverse':
                    t2_transverse_mean = series.mean
                    if t2_transverse_mean > max_t2_transverse_mean:
                        max_t2_transverse_mean = t2_transverse_mean
                        self.t2_transverse_uid = series_uid

    @property
    def study_uid(self):
        return list(self.values())[0].study_uid

    @property
    def t2_sagittal(self) -> Union[None, Series]:
        if self.t2_sagittal_uid is None:
            return None
        else:
            return self[self.t2_sagittal_uid]

    @property
    def t2_transverse(self) -> Union[None, Series]:
        if self.t2_transverse_uid is None:
            return None
        else:
            return self[self.t2_transverse_uid]

    @property
    def t2_sagittal_middle_frame(self) -> Union[None, DICOM]:
        if self.t2_sagittal is None:
            return None
        else:
            return self.t2_sagittal.middle_frame

    def set_t2_sagittal_middle_frame(self, series_uid, instance_uid):
        self.t2_sagittal_uid = series_uid
        self.t2_sagittal.set_middle_frame(instance_uid)

    def t2_transverse_k_nearest(self, pixel_coord, k, size, prob_rotate=0, max_angel=0):
        """

        :param pixel_coord: (M, 2)
        :param k:
        :param size:
        :param prob_rotate:
        :param max_angel:
        :return: (M, k, 1, height, width)
        """
        if self.t2_transverse is None:
            # padding
            return torch.zeros(pixel_coord.shape[0], k, 1, *size)
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord)
        dicoms = self.t2_transverse.k_nearest(human_coord, k)
        images = []
        for series in dicoms:
            temp = []
            for dicom in series:
                image = tf.resize(dicom.image, size)
                if max_angel > 0 and random.random() <= prob_rotate:
                    angel = random.randint(-max_angel, max_angel)
                    image = tf.rotate(image, angel)
                image = tf.to_tensor(image)
                temp.append(image)
            temp = torch.stack(temp, dim=0)
            images.append(temp)
        images = torch.stack(images, dim=0)
        return images


def construct_studies(data_dir, annotation_path=None):
    """
    方便批量构造study的函数
    :param data_dir: 存放study的文件夹
    :param annotation_path: 如果有标注，那么根据标注来确定定位帧
    :return:
    """
    studies = {}
    for study_name in tqdm(os.listdir(data_dir), ascii=True):
        study_dir = os.path.join(data_dir, study_name)
        study = Study(study_dir)
        studies[study.study_uid] = study

    if annotation_path is None:
        return studies
    else:
        counter = {
            't2_sagittal_not_found': [],
            't2_sagittal_miss_match': [],
            't2_sagittal_middle_frame_miss_match': []
        }
        annotation = read_annotation(annotation_path)
        for k in annotation.keys():
            if k[0] in studies:
                study = studies[k[0]]
                if study.t2_sagittal is None:
                    counter['t2_sagittal_not_found'].append(study.study_uid)
                elif study.t2_sagittal_uid != k[1]:
                    counter['t2_sagittal_miss_match'].append(study.study_uid)
                elif study.t2_sagittal_middle_frame.instance_uid != k[2]:
                    counter['t2_sagittal_middle_frame_miss_match'].append(study.study_uid)
                study.set_t2_sagittal_middle_frame(k[1], k[2])
        return studies, annotation, counter
