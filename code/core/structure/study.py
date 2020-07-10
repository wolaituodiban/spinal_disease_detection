import os
import random
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, Union
from tqdm import tqdm
import torch
from torchvision.transforms import functional as tf
from .dicom import DICOM, lazy_property
from .series import Series
from ..data_utils import read_annotation


class Study(dict):
    def __init__(self, study_dir, pool=None):
        dicom_list = []
        if pool is not None:
            async_results = []
            for dicom_name in os.listdir(study_dir):
                dicom_path = os.path.join(study_dir, dicom_name)
                async_results.append(pool.apply_async(DICOM, (dicom_path, )))

            for async_result in async_results:
                async_result.wait()
                dicom = async_result.get()
                dicom_list.append(dicom)
        else:
            for dicom_name in os.listdir(study_dir):
                dicom_path = os.path.join(study_dir, dicom_name)
                dicom = DICOM(dicom_path)
                dicom_list.append(dicom)

        dicom_dict = {}
        for dicom in dicom_list:
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
                if series.plane == 'sagittal' and series.t_type != 'T1':
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

    @lazy_property
    def study_uid(self):
        study_uid_counter = Counter([s.study_uid for s in self.values()])
        return study_uid_counter.most_common(1)[0][0]

    @property
    def t2_sagittal(self) -> Union[None, Series]:
        """
        会被修改的属性不应该lazy
        :return:
        """
        if self.t2_sagittal_uid is None:
            return None
        else:
            return self[self.t2_sagittal_uid]

    @property
    def t2_transverse(self) -> Union[None, Series]:
        """
        会被修改的属性不应该lazy
        :return:
        """
        if self.t2_transverse_uid is None:
            return None
        else:
            return self[self.t2_transverse_uid]

    @property
    def t2_sagittal_middle_frame(self) -> Union[None, DICOM]:
        """
        会被修改的属性不应该lazy
        :return:
        """
        if self.t2_sagittal is None:
            return None
        else:
            return self.t2_sagittal.middle_frame

    def set_t2_sagittal_middle_frame(self, series_uid, instance_uid):
        assert series_uid in self
        self.t2_sagittal_uid = series_uid
        self.t2_sagittal.set_middle_frame(instance_uid)

    def t2_transverse_k_nearest(self, pixel_coord, k, size, max_dist, prob_rotate=0,
                                max_angel=0) -> (torch.Tensor, torch.Tensor):
        """

        :param pixel_coord: (M, 2)
        :param k:
        :param size:
        :param max_dist:
        :param prob_rotate:
        :param max_angel:
        :return: 图像张量(M, k, 1, height, width)，masks(M， k)
            masks: 为None的位置将被标注为True
        """
        if k <= 0 or self.t2_transverse is None:
            # padding
            images = torch.zeros(pixel_coord.shape[0], k, 1, *size)
            masks = torch.zeros(*images.shape[:2], dtype=torch.bool)
            return images, masks
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord)
        dicoms = self.t2_transverse.k_nearest(human_coord, k, max_dist)
        images = []
        masks = []
        for point, series in zip(human_coord, dicoms):
            temp_images = []
            temp_masks = []
            for dicom in series:
                if dicom is None:
                    temp_masks.append(True)
                    image = torch.zeros(1, *size)
                else:
                    temp_masks.append(False)
                    projection = dicom.projection(point)
                    image, projection = dicom.transform(
                        projection, size=[size[0]*2, size[1]*2], prob_rotate=prob_rotate, max_angel=max_angel,
                        tensor=False
                    )
                    image = tf.crop(
                        image, int(projection[0]-size[0]//2), int(projection[1]-size[1]//2), size[0], size[1])
                    image = tf.to_tensor(image)
                temp_images.append(image)
            temp_images = torch.stack(temp_images, dim=0)
            images.append(temp_images)
            masks.append(temp_masks)
        images = torch.stack(images, dim=0)
        masks = torch.tensor(masks, dtype=torch.bool)
        return images, masks

    def transform(self, v_coords, d_coords, transverse_size, sagittal_size=None, k_nearest=0,
                  max_dist=6, prob_rotate=0, max_angel=0, sagittal_shift=0):
        """
        生成study的训练数据
        :param v_coords:
        :param d_coords:
        :param transverse_size:
        :param sagittal_size:
        :param k_nearest:
        :param max_dist:
        :param prob_rotate:
        :param max_angel:
        :param sagittal_shift:
        :return:
        """
        # 随机对中间帧进行偏移操作
        dicom_idx = self.t2_sagittal.instance_uids[self.t2_sagittal_middle_frame.instance_uid]
        dicom_idx = random.randint(dicom_idx-sagittal_shift, dicom_idx+sagittal_shift)
        dicom: DICOM = self.t2_sagittal[dicom_idx]
        # 计算原先的坐标在偏移后的帧上的投影坐标
        pixel_coord = torch.cat([v_coords, d_coords], dim=0)
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord)
        pixel_coord = dicom.projection(human_coord)
        # 将被选中的矢状图转换成张量
        sagittal_image, pixel_coord, distmap = dicom.transform(
            pixel_coord, sagittal_size, prob_rotate, max_angel, distmap=True)

        # 因为锥体的轴状图太少了，所以只提取椎间盘的轴状图
        transverse_image, t_mask = self.t2_transverse_k_nearest(
            d_coords, k=k_nearest, size=transverse_size, max_dist=max_dist,
            prob_rotate=prob_rotate, max_angel=max_angel
        )
        return transverse_image, sagittal_image, distmap, pixel_coord, t_mask


def _construct_studies(data_dir, multiprocessing=False):
    studies: Dict[str, Study] = {}
    if multiprocessing:
        pool = Pool(cpu_count())
    else:
        pool = None

    for study_name in tqdm(os.listdir(data_dir), ascii=True):
        study_dir = os.path.join(data_dir, study_name)
        study = Study(study_dir, pool)
        studies[study.study_uid] = study

    if pool is not None:
        pool.close()
        pool.join()
    return studies


def count_error_middle_frame(studies, annotation):
    counter = {
        't2_sagittal_not_found': [],
        't2_sagittal_miss_match': [],
        't2_sagittal_middle_frame_miss_match': []
    }
    for k in annotation.keys():
        if k[0] in studies:
            study = studies[k[0]]
            if study.t2_sagittal is None:
                counter['t2_sagittal_not_found'].append(study.study_uid)
            elif study.t2_sagittal_uid != k[1]:
                counter['t2_sagittal_miss_match'].append(study.study_uid)
            else:
                t2_sagittal = study.t2_sagittal
                gt_z_index = t2_sagittal.instance_uids[k[2]]
                middle_frame = t2_sagittal.middle_frame
                z_index = t2_sagittal.instance_uids[middle_frame.instance_uid]
                if abs(gt_z_index - z_index) > 1:
                    counter['t2_sagittal_middle_frame_miss_match'].append(study.study_uid)
    return counter


def set_middle_frame(studies: Dict[str, Study], annotation):
    for k in annotation.keys():
        if k[0] in studies:
            study = studies[k[0]]
            study.set_t2_sagittal_middle_frame(k[1], k[2])


def construct_studies(data_dir, annotation_path=None, multiprocessing=False):
    """
    方便批量构造study的函数
    :param data_dir: 存放study的文件夹
    :param multiprocessing:
    :param annotation_path: 如果有标注，那么根据标注来确定定位帧
    :return:
    """
    studies = _construct_studies(data_dir, multiprocessing)

    # 使用annotation制定正确的中间帧
    if annotation_path is None:
        return studies
    else:
        annotation = read_annotation(annotation_path)
        counter = count_error_middle_frame(studies, annotation)
        set_middle_frame(studies, annotation)
        return studies, annotation, counter
