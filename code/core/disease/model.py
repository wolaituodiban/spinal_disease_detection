from copy import deepcopy
from typing import Tuple
import torch
import torchvision.transforms.functional as tf
from .loss import DisLoss
from ..structure import Study
from ..key_point import extract_point_feature, KeyPointModel
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_VERTEBRA_DISEASE_ID, SPINAL_DISC_ID, SPINAL_DISC_DISEASE_ID


VERTEBRA_POINT_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_ID.items()}
VERTEBRA_DISEASE_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_DISEASE_ID.items()}
DISC_POINT_INT2STR = {v: k for k, v in SPINAL_DISC_ID.items()}
DISC_DISEASE_INT2STR = {v: k for k, v in SPINAL_DISC_DISEASE_ID.items()}


class DiseaseModelBase(torch.nn.Module):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 num_vertebra_diseases=len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_diseases=len(SPINAL_DISC_DISEASE_ID)):
        super().__init__()
        self.sagittal_size = sagittal_size
        self.num_vertebra_diseases = num_vertebra_diseases
        self.num_disc_disease = num_disc_diseases
        self.backbone = deepcopy(kp_model)

    @staticmethod
    def _gen_annotation(study: Study, vertebra_coords, vertebra_scores, disc_coords, disc_scores) -> dict:
        """

        :param study:
        :param vertebra_coords: Nx2
        :param vertebra_scores: V
        :param disc_scores: Dx1
        :return:
        """
        z_index = study.t2_sagittal.instance_uids[study.t2_sagittal_middle_frame.instance_uid]
        point = []
        for i, (coord, score) in enumerate(zip(vertebra_coords, vertebra_scores)):
            vertebra = int(torch.argmax(score, dim=-1).cpu())
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': VERTEBRA_POINT_INT2STR[i],
                    'vertebra': VERTEBRA_DISEASE_INT2STR[vertebra]
                },
                'zIndex': z_index
            })
        for i, (coord, score) in enumerate(zip(disc_coords, disc_scores)):
            disc = int(torch.argmax(score, dim=-1).cpu())
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': DISC_POINT_INT2STR[i],
                    'disc': DISC_DISEASE_INT2STR[disc]
                },
                'zIndex': z_index
            })
        annotation = {
            'studyUid': study.study_uid,
            'data': [
                {
                    'instanceUid': study.t2_sagittal_middle_frame.instance_uid,
                    'seriesUid': study.t2_sagittal_middle_frame.series_uid,
                    'annotation': [
                        {
                            'data': {
                                'point': point,
                            }
                        }
                    ]
                }
            ]
        }
        return annotation

    def forward(self, study: Study, to_dict=False):
        kp_frame = study.t2_sagittal_middle_frame
        # 将图片放缩到模型设定的大小
        sagittal = tf.resize(kp_frame.image, self.sagittal_size)
        sagittal = tf.to_tensor(sagittal).unsqueeze(0)

        v_coord, d_coord, _, feature_maps = self.backbone(sagittal, return_more=True)

        # 将预测的坐标调整到原来的大小，注意要在extract_point_feature之后变换
        height_ratio = self.sagittal_size[0] / kp_frame.size[1]
        width_ratio = self.sagittal_size[1] / kp_frame.size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=v_coord.device)
        v_coord = (v_coord.float() / ratio).round()[0]
        d_coord = (d_coord.float() / ratio).round()[0]

        v_score = torch.zeros(v_coord.shape[0], self.num_vertebra_diseases)
        v_score[:, 1] = 1

        d_score = torch.zeros(d_coord.shape[0], self.num_disc_disease)
        d_score[:, 0] = 1

        if to_dict:
            return self._gen_annotation(study, v_coord, v_score, d_coord, d_score)
        else:
            return v_coord, v_score, d_coord, d_score


class DiseaseModel(DiseaseModelBase):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 num_vertebra_diseases=len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_diseases=len(SPINAL_DISC_DISEASE_ID),
                 share_backbone=False,
                 vertebra_loss=DisLoss([2.2727, 0.6410]),
                 disc_loss=DisLoss([0.4327, 0.7930, 0.8257, 6.4286, 16.3636]),
                 loss_scaler=1,
                 use_kp_loss=False,
                 dropout=0):
        super(DiseaseModel, self).__init__(kp_model, sagittal_size, num_vertebra_diseases, num_disc_diseases)
        if share_backbone:
            self.kp_model = None
        else:
            self.kp_model = kp_model

        self.vertebra_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(self.backbone.out_channels, num_vertebra_diseases)
        )

        self.disc_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(self.backbone.out_channels, num_disc_diseases)
        )
        self.use_kp_loss = use_kp_loss
        self.vertebra_loss = vertebra_loss
        self.disc_loss = disc_loss
        self.loss_scaler = loss_scaler

    # def crop(self, image, point):
    #     left, right = point[1] - self.crop_size, point[1] + self.crop_size
    #     top, bottom = point[0] - self.crop_size, point[0] + self.crop_size
    #     return image[:, left:right, top:bottom]
    #
    # def forward(self, sagittals, transverses, v_labels, d_labels, distmaps):
    #     v_patches = []
    #     for sagittal, v_label in zip(sagittals, v_labels):
    #         for point in v_label:
    #             patch = self.crop(sagittal, point)
    #             v_patches.append(patch)
    #
    #     d_patches = []
    #     for sagittal, d_label in zip(sagittals, d_labels):
    #         for point in d_label:
    #             patch = self.crop(sagittal, point)
    #             d_patches.append(patch)
    #     return v_patches, d_patches

    def _train(self, sagittals, transverses, distmaps, v_labels, d_labels, v_masks, d_masks) -> tuple:
        if self.use_kp_loss:
            masks = torch.cat([v_masks, d_masks], dim=-1)
        else:
            distmaps = None
            masks = None

        kp_loss, _, feature_maps = self.backbone(sagittals, distmaps, masks, return_more=True)
        v_features = extract_point_feature(feature_maps, v_labels, *sagittals.shape[-2:])
        d_features = extract_point_feature(feature_maps, d_labels, *sagittals.shape[-2:])

        v_scores = self.vertebra_head(v_features)
        d_scores = self.disc_head(d_features)

        v_loss = self.vertebra_loss(v_scores, v_labels[:, :, -1], v_masks) * self.loss_scaler
        d_loss = self.disc_loss(d_scores, d_labels[:, :, -1], d_masks) * self.loss_scaler

        loss = torch.stack([v_loss, d_loss])
        if kp_loss is None:
            return loss,
        elif len(kp_loss.shape) > 0:
            return torch.cat([kp_loss.flatten(), loss], dim=0),
        else:
            return torch.stack([kp_loss.unsqueeze(0), loss], dim=0),

    def _inference(self, study: Study, to_dict=False):
        kp_frame = study.t2_sagittal_middle_frame
        # 将图片放缩到模型设定的大小
        sagittal = tf.resize(kp_frame.image, self.sagittal_size)
        sagittal = tf.to_tensor(sagittal).unsqueeze(0)

        v_coord, d_coord, _, feature_maps = self.backbone(sagittal, return_more=True)
        if self.kp_model is not None:
            v_coord, d_coord = self.kp_model(sagittal)
        v_feature = extract_point_feature(feature_maps, v_coord, *self.sagittal_size)
        d_feature = extract_point_feature(feature_maps, d_coord, *self.sagittal_size)

        v_score = self.vertebra_head(v_feature)[0]
        d_score = self.disc_head(d_feature)[0]

        # 将预测的坐标调整到原来的大小，注意要在extract_point_feature之后变换
        height_ratio = self.sagittal_size[0] / kp_frame.size[1]
        width_ratio = self.sagittal_size[1] / kp_frame.size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=v_coord.device)
        v_coord = (v_coord.float() / ratio).round()[0]
        d_coord = (d_coord.float() / ratio).round()[0]

        if to_dict:
            return self._gen_annotation(study, v_coord, v_score, d_coord, d_score)
        else:
            return v_coord, v_score, d_coord, d_score

    def forward(self, *args, **kwargs):
        if self.training:
            return self._train(*args, **kwargs)
        else:
            return self._inference(*args, **kwargs)
