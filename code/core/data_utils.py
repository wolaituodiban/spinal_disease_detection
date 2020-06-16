import json
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from .dicom_utils import read_one_dcm


def read_dcms(dcm_dir, error_msg=False) -> (Dict[Tuple[str, str, str], Image.Image], Dict[Tuple[str, str, str], dict]):
    """
    读取文件夹内的所有dcm文件
    :param dcm_dir:
    :param error_msg: 是否打印错误信息
    :return: 包含图像信息的字典，和包含元数据的字典
    """
    dcm_paths = []
    for study in os.listdir(dcm_dir):
        study_path = os.path.join(dcm_dir, study)
        for dcm_name in os.listdir(study_path):
            dcm_path = os.path.join(study_path, dcm_name)
            dcm_paths.append(dcm_path)

    with Pool(cpu_count()) as pool:
        async_results = []
        for dcm_path in dcm_paths:
            async_results.append(pool.apply_async(read_one_dcm, (dcm_path,)))

        images, metainfos = {}, {}
        for async_result in tqdm(async_results, ascii=True):
            async_result.wait()
            try:
                metainfo, image = async_result.get()
            except RuntimeError as e:
                if error_msg:
                    print(e)
                continue
            key = metainfo['studyUid'], metainfo['seriesUid'], metainfo['instanceUid']
            del metainfo['studyUid'], metainfo['seriesUid'], metainfo['instanceUid']
            images[key] = tf.to_pil_image(image)
            metainfos[key] = metainfo

    return images, metainfos


def get_spacing(metainfos: Dict[Tuple[str, str, str], dict]) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """
    从元数据中获取像素点间距的信息
    :param metainfos:
    :return:
    """
    output = {}
    for k, v in metainfos.items():
        spacing = v['pixelSpacing']
        spacing = spacing.split('\\')
        spacing = list(map(float, spacing))
        output[k] = torch.tensor(spacing)
    return output


with open(os.path.join(os.path.dirname(__file__), 'json_files/spinal_vertebra_id.json'), 'r') as file:
    SPINAL_VERTEBRA_ID = json.load(file)

with open(os.path.join(os.path.dirname(__file__), 'json_files/spinal_disc_id.json'), 'r') as file:
    SPINAL_DISC_ID = json.load(file)

assert set(SPINAL_VERTEBRA_ID.keys()).isdisjoint(set(SPINAL_DISC_ID.keys()))

with open(os.path.join(os.path.dirname(__file__), 'json_files/spinal_vertebra_disease.json'), 'r') as file:
    SPINAL_VERTEBRA_DISEASE_ID = json.load(file)

with open(os.path.join(os.path.dirname(__file__), 'json_files/spinal_disc_disease.json'), 'r') as file:
    SPINAL_DISC_DISEASE_ID = json.load(file)


def read_annotation(path) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
    """

    :param path:
    :return: 字典的key是（studyUid，seriesUid，instance_uid）
             字典的value是两个矩阵，第一个矩阵对应锥体，第一个矩阵对应椎间盘
             矩阵每一行对应一个脊柱的位置，前两列是位置的坐标(横坐标, 纵坐标)，之后每一列对应一种疾病
             坐标为0代表缺失
             ！注意图片的坐标和tensor的坐标是转置关系的
    """
    with open(path, 'r') as annotation_file:
        # non_hit_count用来统计为被编码的标记的数量，用于预警
        non_hit_count = {}
        annotation = {}
        for x in json.load(annotation_file):
            study_uid = x['studyUid']

            assert len(x['data']) == 1, (study_uid, len(x['data']))
            data = x['data'][0]
            instance_uid = data['instanceUid']
            series_uid = data['seriesUid']

            assert len(data['annotation']) == 1, (study_uid, len(data['annotation']))
            points = data['annotation'][0]['data']['point']

            vertebra_label = torch.zeros([len(SPINAL_VERTEBRA_ID), 2+len(SPINAL_VERTEBRA_DISEASE_ID)], dtype=torch.long)
            disc_label = torch.zeros([len(SPINAL_DISC_ID), 2+len(SPINAL_DISC_DISEASE_ID)], dtype=torch.long)
            for point in points:
                identification = point['tag']['identification']
                if identification in SPINAL_VERTEBRA_ID:
                    position = SPINAL_VERTEBRA_ID[identification]
                    diseases = point['tag']['vertebra']

                    vertebra_label[position, :2] = torch.tensor(point['coord'])
                    for disease in diseases.split(','):
                        disease = SPINAL_VERTEBRA_DISEASE_ID[disease]
                        vertebra_label[position, 2+disease] = 1
                elif identification in SPINAL_DISC_ID:
                    position = SPINAL_DISC_ID[identification]
                    diseases = point['tag']['disc']

                    disc_label[position, :2] = torch.tensor(point['coord'])
                    for disease in diseases.split(','):
                        if disease not in SPINAL_DISC_DISEASE_ID:
                            print(study_uid, '\n', series_uid, '\n', instance_uid, '\n', point)
                        else:
                            disease = SPINAL_DISC_DISEASE_ID[disease]
                            disc_label[position, 2+disease] = 1
                elif identification in non_hit_count:
                    non_hit_count[identification] += 1
                else:
                    non_hit_count[identification] = 1

            annotation[study_uid, series_uid, instance_uid] = vertebra_label, disc_label
    if len(non_hit_count) > 0:
        print(non_hit_count)
    return annotation


def resize(size: Tuple[int, int], image: Image.Image, spacing: torch.Tensor, *annotation: torch.Tensor):
    """

    :param size: [height, width]
    :param image: 图像
    :param spacing: 像素点间距
    :param annotation: 标注
    :return: resize之后的image，spacing，annotation
    """
    height_ratio = size[0] / image.size[0]
    width_ratio = size[1] / image.size[1]

    ratio = torch.tensor([width_ratio, height_ratio])
    spacing = spacing * ratio
    annotation = [_.clone().float() for _ in annotation]
    for _ in annotation:
        _[:, :2] *= ratio
    image = tf.resize(image, size)
    return image, spacing, *annotation


def gen_label(image: torch.Tensor, spacing: torch.Tensor, *annotation: torch.Tensor) -> List[torch.Tensor]:
    """
    计算每个像素点到标注像素点的物理距离
    :param image:
    :param annotation:
    :param spacing:
    :return:
    """
    coord = torch.where(image.squeeze() < np.inf)
    # 注意需要反转横纵坐标
    coord = torch.stack(coord[::-1], dim=1).reshape(image.size(1), image.size(2), 2)
    dists = []
    for _ in annotation:
        dist = []
        for point in _:
            dist.append((((coord - point[:2]) * spacing) ** 2).sum(dim=-1).sqrt())
        dist = torch.stack(dist, dim=0)
        dists.append(dist)
    return dists
