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
                 sagittal_size: Tuple[int, int]):
        super().__init__()
        self.sagittal_size = sagittal_size
        self.num_vertebra_diseases = len(SPINAL_VERTEBRA_DISEASE_ID)
        self.num_disc_diseases = len(SPINAL_DISC_DISEASE_ID)
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

        d_score = torch.zeros(d_coord.shape[0], self.num_disc_diseases)
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


class DiseaseModel(DiseaseModelBase):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 sagittal_shift: int = 0,
                 share_backbone=False,
                 vertebra_loss=DisLoss([2.2727, 0.6410]),
                 disc_loss=DisLoss([0.4327, 0.7930, 0.8257, 6.4286, 16.3636]),
                 loss_scaler=1,
                 use_kp_loss=False,
                 max_dist=6):
        super().__init__(kp_model=kp_model, sagittal_size=sagittal_size)
        if share_backbone:
            self.kp_model = None
        else:
            self.kp_model = kp_model

        self.vertebra_head = torch.nn.Linear(self.out_channels, self.num_vertebra_diseases)
        self.disc_head = torch.nn.Linear(self.out_channels, self.num_disc_diseases)

        self.sagittal_shift = sagittal_shift

        self.use_kp_loss = use_kp_loss
        self.vertebra_loss = vertebra_loss
        self.disc_loss = disc_loss
        self.loss_scaler = loss_scaler
        self.max_dist = max_dist

        # 为了兼容性，实际上没有用
        self.k_nearest = 0
        self.transverse_size = self.sagittal_size

    def disease_parameters(self, recurse=True):
        for p in self.vertebra_head.parameters(recurse):
            yield p
        for p in self.disc_head.parameters(recurse):
            yield p

    def _mask_pred(self, pred_coords, distmaps):
        width_indices = pred_coords[:, :, 0].flatten().clamp(0, distmaps.shape[-1]-1)
        height_indices = pred_coords[:, :, 1].flatten().clamp(0, distmaps.shape[-2]-1)

        image_indices = torch.arange(pred_coords.shape[0], device=pred_coords.device)
        image_indices = image_indices.unsqueeze(1).expand(-1, pred_coords.shape[1]).flatten()
        point_indices = torch.arange(pred_coords.shape[1], device=pred_coords.device).repeat(pred_coords.shape[0])

        new_masks = distmaps[image_indices, point_indices, height_indices, width_indices] < self.max_dist
        new_masks = new_masks.reshape(pred_coords.shape[0], -1)
        return new_masks

    def _adjust_pred(self, pred_coords, distmaps, gt_coords):
        gt_coords = gt_coords.to(pred_coords.device)
        new_masks = self._mask_pred(pred_coords, distmaps)
        new_masks = torch.bitwise_not(new_masks)
        pred_coords[new_masks] = gt_coords[new_masks]
        return pred_coords

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

        # 决定使用那种坐标训练
        if self.kp_model is not None:
            v_coords, d_coords = self.kp_model.eval()(sagittals)

        # 将错误的预测改为正确位置
        v_coords = self._adjust_pred(
            v_coords, distmaps[:, :self.num_vertebra_points], v_labels[:, :, :2]
        )
        d_coords = self._adjust_pred(
            d_coords, distmaps[:, self.num_vertebra_points:], d_coords[:, :, :2]
        )

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
        middle_frame_size = study.t2_sagittal_middle_frame.size
        middle_frame_uid = study.t2_sagittal_middle_frame.instance_uid
        middle_frame_idx = study.t2_sagittal.instance_uids[middle_frame_uid]
        # 传入三张矢量图
        sagittal_dicoms = []
        for idx in range(middle_frame_idx - self.sagittal_shift, middle_frame_idx + self.sagittal_shift + 1):
            sagittal_dicoms.append(study.t2_sagittal[idx])

        sagittal_images = []
        for dicom in sagittal_dicoms:
            # 将图片放缩到模型设定的大小
            image = tf.resize(dicom.image, self.sagittal_size)
            image = tf.to_tensor(image)
            sagittal_images.append(image)
        sagittal_images = torch.stack(sagittal_images, dim=0)

        if self.kp_model is not None:
            v_coord, d_coord = self.kp_model(sagittal_images)
            _, feature_maps = self.backbone.cal_scores(sagittal_images)
        else:
            v_coord, d_coord, _, feature_maps = self.backbone(sagittal_images, return_more=True)

        # 修正坐标
        # 具体逻辑是先将多张矢状图上预测的点坐标都投影到中间帧上
        # 然后在中间帧上求中位数
        # 最后将中位数坐标分别投影回多张矢状图上，并以此提取点特征
        v_coord_human = [dicom.pixel_coord2human_coord(coord) for dicom, coord in zip(sagittal_dicoms, v_coord)]
        d_coord_human = [dicom.pixel_coord2human_coord(coord) for dicom, coord in zip(sagittal_dicoms, d_coord)]

        v_coord = torch.stack([study.t2_sagittal_middle_frame.projection(coord) for coord in v_coord_human], dim=0)
        d_coord = torch.stack([study.t2_sagittal_middle_frame.projection(coord) for coord in d_coord_human], dim=0)

        v_coord_med = v_coord.median(dim=0)[0]
        d_coord_med = d_coord.median(dim=0)[0]

        v_coord_med_human = study.t2_sagittal_middle_frame.pixel_coord2human_coord(v_coord_med)
        d_coord_med_human = study.t2_sagittal_middle_frame.pixel_coord2human_coord(d_coord_med)

        v_coord = torch.stack([dicom.projection(v_coord_med_human) for dicom in sagittal_dicoms], dim=0)
        d_coord = torch.stack([dicom.projection(d_coord_med_human) for dicom in sagittal_dicoms], dim=0)

        # 在三个feature_map上一起在提取点特征
        v_feature = extract_point_feature(feature_maps, v_coord, *self.sagittal_size)
        d_feature = extract_point_feature(feature_maps, d_coord, *self.sagittal_size)

        # 提取transverse特征
        transverse, t_mask = study.t2_transverse_k_nearest(
            d_coord_med[0].cpu(), k=self.k_nearest, size=self.transverse_size, max_dist=self.max_dist
        )
        num_sagittals = 2 * self.sagittal_shift + 1
        d_feature = self._agg_features(
            d_feature,
            transverse.unsqueeze(0).expand(num_sagittals, -1, -1, 1, -1, -1),
            t_mask.unsqueeze(0).expand(num_sagittals, -1, -1)
        )

        # 计算分数，并取中位数
        v_score = self.vertebra_head(v_feature).median(dim=0)[0]
        d_score = self.disc_head(d_feature).median(dim=0)[0]

        # 将预测的坐标调整到原来的大小，注意要在extract_point_feature之后变换
        height_ratio = self.sagittal_size[0] / middle_frame_size[1]
        width_ratio = self.sagittal_size[1] / middle_frame_size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=v_coord_med.device)

        # 将坐标变回原来的大小
        v_coord_med = (v_coord_med.float() / ratio).round()
        d_coord_med = (d_coord_med.float() / ratio).round()

        if to_dict:
            return self._gen_annotation(study, v_coord_med, v_score, d_coord_med, d_score)
        else:
            return v_coord_med, v_score, d_coord_med, d_score


class DiseaseModelV2(DiseaseModel):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 transverse_size: Tuple[int, int],
                 sagittal_shift: int = 0,
                 share_backbone=False,
                 vertebra_loss=DisLoss([2.2727, 0.6410]),
                 disc_loss=DisLoss([0.4327, 0.7930, 0.8257, 6.4286, 16.3636]),
                 loss_scaler=1,
                 use_kp_loss=False,
                 max_dist=6,
                 k_nearest=0,
                 nhead=8,
                 transverse_only=False,
                 ):
        super().__init__(kp_model=kp_model, sagittal_size=sagittal_size, sagittal_shift=sagittal_shift,
                         share_backbone=share_backbone, vertebra_loss=vertebra_loss, disc_loss=disc_loss,
                         loss_scaler=loss_scaler, use_kp_loss=use_kp_loss, max_dist=max_dist)
        self.transverse_size = transverse_size
        self.k_nearest = k_nearest
        self.transverse_only = transverse_only

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
        :param d_point_feats: 椎间盘点特征，(num_batch, num_points, out_channels)
        :param transverses: (num_batch, num_points, k_nearest, 1, height, width)
        :param t_masks: (num_batch, num_points, k_nearest)，轴状图为padding的地方是True
        :return:
        """
        # T当k nearest为0时，功能退化为V1
        if self.k_nearest <= 0:
            return d_point_feats

        t_features = self.backbone.cal_backbone(transverses.flatten(end_dim=2))
        t_features = self.transverse_block(t_features).reshape(*transverses.shape[:3], -1)
        t_masks = t_masks.to(t_features.device)

        if self.transverse_only:
            all_features = t_features
            all_masks = torch.zeros(*t_masks.shape[:2], 1, device=t_masks.device, dtype=t_masks.dtype)
            all_masks = torch.cat([all_masks, t_masks[:, :, 1:]], dim=2)
        else:
            all_features = torch.cat([d_point_feats.unsqueeze(2), t_features], dim=2)
            # 矢状图的特征是全部都要用上的，所以d_masks全为False
            d_masks = torch.zeros(*t_masks.shape[:2], 1, device=t_masks.device, dtype=t_masks.dtype)
            all_masks = torch.cat([d_masks, t_masks], dim=2)

        all_features = all_features.flatten(end_dim=1).permute(1, 0, 2)
        all_masks = all_masks.flatten(end_dim=1)

        final_features = self.aggregation(all_features, src_key_padding_mask=all_masks)[0]
        final_features = final_features.reshape(*transverses.shape[:2], -1)
        return final_features
