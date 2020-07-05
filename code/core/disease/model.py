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

    @property
    def out_channels(self):
        return self.backbone.out_channels

    @property
    def num_vertebra_points(self):
        return self.backbone.num_vertebra_points

    @property
    def num_disc_points(self):
        return self.backbone.num_disc_point

    @property
    def kp_parameters(self):
        return self.backbone.kp_parameters

    @property
    def resnet_out_channels(self):
        return self.backbone.resnet_out_channels

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

    def _train(self, sagittals, _, distmaps, v_labels, d_labels, v_masks, d_masks, t_masks) -> tuple:
        masks = torch.cat([v_masks, d_masks], dim=-1)
        return self.backbone(sagittals, distmaps, masks)

    def _inference(self, study: Study, to_dict=False):
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

    def forward(self, *args, **kwargs):
        if self.training:
            return self._train(*args, **kwargs)
        else:
            return self._inference(*args, **kwargs)


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
                 kp_max_dist=6):
        super().__init__(kp_model=kp_model, sagittal_size=sagittal_size,
                         num_vertebra_diseases=num_vertebra_diseases, num_disc_diseases=num_disc_diseases)
        if share_backbone:
            self.kp_model = None
        else:
            self.kp_model = kp_model

        self.vertebra_head = torch.nn.Linear(self.out_channels, num_vertebra_diseases)
        self.disc_head = torch.nn.Linear(self.out_channels, num_disc_diseases)

        self.use_kp_loss = use_kp_loss
        self.vertebra_loss = vertebra_loss
        self.disc_loss = disc_loss
        self.loss_scaler = loss_scaler
        self.kp_max_dist = kp_max_dist

        # 为了兼容性，实际上没有用
        self.k_nearest = 0
        self.transverse_size = self.sagittal_size
        self.transverse_max_dist = kp_max_dist

    def disease_parameters(self, recurse=True):
        for p in self.vertebra_head.parameters(recurse):
            yield p
        for p in self.disc_head.parameters(recurse):
            yield p

    # def crop(self, image, point):
    #     left, right = point[0] - self.crop_size, point[0] + self.crop_size
    #     top, bottom = point[1] - self.crop_size, point[1] + self.crop_size
    #     return image[:, top:bottom, left:right]
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

    def _adjust_masks(self, pred_coords, distmaps, masks):
        """
        将距离大于阈值的预测坐标的mask变成false
        :param pred_coords: (num_batch, num_points, 2)
        :param distmaps: (num_batch, num_points, height, width)
        :param masks: (num_batch, num_points)
        :return:
        """
        if self.kp_max_dist <= 0:
            return masks

        width_indices = pred_coords[:, :, 0].flatten().clamp(0, distmaps.shape[-1]-1)
        height_indices = pred_coords[:, :, 1].flatten().clamp(0, distmaps.shape[-2]-1)

        image_indices = torch.arange(pred_coords.shape[0], device=pred_coords.device)
        image_indices = image_indices.unsqueeze(1).expand(-1, pred_coords.shape[1]).flatten()
        point_indices = torch.arange(pred_coords.shape[1], device=pred_coords.device).repeat(pred_coords.shape[0])

        new_masks = distmaps[image_indices, point_indices, height_indices, width_indices] < self.kp_max_dist
        new_masks = new_masks.reshape(pred_coords.shape[0], -1)

        # 且运算
        new_masks = torch.bitwise_and(new_masks, masks)
        return new_masks

    @staticmethod
    def _agg_features(d_point_feats, transverse, t_masks):
        """
        未来兼容，融合椎间盘矢状图和轴状图的特征
        :param d_point_feats:
        :param transverse:
        :param t_masks
        :return:
        """
        return d_point_feats

    def _train(self, sagittals, transverse, distmaps, v_labels, d_labels, v_masks, d_masks, t_masks) -> tuple:
        if self.use_kp_loss:
            masks = torch.cat([v_masks, d_masks], dim=-1)
            kp_loss, v_coords, d_coords, _, feature_maps = self.backbone(
                sagittals, distmaps, masks, return_more=True)
        else:
            kp_loss, v_coords, d_coords, _, feature_maps = self.backbone(
                sagittals, None, None, return_more=True)

        #
        if self.loss_scaler <= 0:
            return kp_loss,

        # 用于单独训练disease heads
        if self.kp_model is not None:
            v_coords, d_coords = self.kp_model.eval()(sagittals)

        # 挑选正确的预测点
        v_masks = self._adjust_masks(
            v_coords, distmaps[:, :self.num_vertebra_points], v_masks)
        d_masks = self._adjust_masks(
            d_coords, distmaps[:, self.num_vertebra_points:], d_masks)

        # 提取坐标点上的特征
        v_features = extract_point_feature(feature_maps, v_coords, *sagittals.shape[-2:])
        d_features = extract_point_feature(feature_maps, d_coords, *sagittals.shape[-2:])

        # 提取transverse特征
        d_features = self._agg_features(d_features, transverse, t_masks)

        # 计算scores
        v_scores = self.vertebra_head(v_features)
        d_scores = self.disc_head(d_features)

        # 计算损失
        v_loss = self.vertebra_loss(v_scores, v_labels[:, :, -1], v_masks)
        d_loss = self.disc_loss(d_scores, d_labels[:, :, -1], d_masks)

        loss = torch.stack([v_loss, d_loss]) * self.loss_scaler
        if kp_loss is None:
            return loss,
        elif len(kp_loss.shape) > 0:
            return torch.cat([kp_loss.flatten(), loss], dim=0),
        else:
            return torch.cat([kp_loss.unsqueeze(0), loss], dim=0),

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

        # 提取transverse特征
        transverse, t_masks = study.t2_transverse_k_nearest(
            d_coord[0].cpu(), k=self.k_nearest, size=self.transverse_size, max_dist=self.transverse_max_dist
        )
        d_feature = self._agg_features(d_feature, transverse.unsqueeze(0), t_masks.unsqueeze(0))

        # 计算分数
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


class DiseaseModelV2(DiseaseModel):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 transverse_size: Tuple[int, int],
                 num_vertebra_diseases=len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_diseases=len(SPINAL_DISC_DISEASE_ID),
                 share_backbone=False,
                 vertebra_loss=DisLoss([2.2727, 0.6410]),
                 disc_loss=DisLoss([0.4327, 0.7930, 0.8257, 6.4286, 16.3636]),
                 loss_scaler=1,
                 use_kp_loss=False,
                 kp_max_dist=6,
                 transverse_max_dist=8,
                 k_nearest=1,
                 nhead=8):
        super().__init__(kp_model=kp_model, sagittal_size=sagittal_size, num_vertebra_diseases=num_vertebra_diseases,
                         num_disc_diseases=num_disc_diseases, share_backbone=share_backbone,
                         vertebra_loss=vertebra_loss, disc_loss=disc_loss, loss_scaler=loss_scaler,
                         use_kp_loss=use_kp_loss, kp_max_dist=kp_max_dist)
        self.transverse_size = transverse_size
        self.transverse_max_dist = transverse_max_dist
        self.k_nearest = k_nearest

        self.transverse_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.resnet_out_channels, self.out_channels,
                kernel_size=(transverse_size[0]//32, transverse_size[1]//32)),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU(inplace=True)
        )
        self.aggregation = torch.nn.TransformerEncoderLayer(self.out_channels, nhead=nhead)

    def disease_parameters(self, recurse=True):
        for p in super(DiseaseModelV2, self).disease_parameters(recurse):
            yield p
        for p in self.transverse_block.parameters(recurse):
            yield p
        for p in self.aggregation.parameters(recurse):
            yield p

    def _agg_features(self, d_point_feats, transverses, t_masks):
        """
        融合椎间盘的矢状图和轴状图的特征
        :param d_point_feats: (num_batch, num_points, out_channels)
        :param transverses: (num_batch, num_points, k_nearest, 1, height, width)
        :param t_masks: (num_batch, num_points, k_nearest)，轴状图为padding的地方是True
        :return:
        """
        t_features = self.backbone.cal_backbone(transverses.flatten(end_dim=2))
        t_features = self.transverse_block(t_features).reshape(*transverses.shape[:3], -1)

        all_features = torch.cat([d_point_feats.unsqueeze(2), t_features], dim=2)
        all_features = all_features.flatten(end_dim=1).permute(1, 0, 2)

        t_masks = t_masks.to(all_features.device)
        # 矢状图的特征是全部都要用上的，所以s_masks全为False
        s_masks = torch.zeros(*t_masks.shape[:2], 1, device=t_masks.device, dtype=t_masks.dtype)
        all_masks = torch.cat([s_masks, t_masks], dim=-1)
        all_masks = all_masks.flatten(end_dim=1)

        final_features = self.aggregation(all_features, src_key_padding_mask=all_masks)[0]
        final_features = final_features.reshape(*transverses.shape[:2], -1)
        return final_features
