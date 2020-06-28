import torch
import torchvision.transforms.functional as tf
from .loss import DisLoss
from ..structure import Study
from ..key_point import KeyPointModel
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_VERTEBRA_DISEASE_ID, SPINAL_DISC_ID, SPINAL_DISC_DISEASE_ID
from ..data_utils import PADDING_VALUE


VERTEBRA_POINT_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_ID.items()}
VERTEBRA_DISEASE_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_DISEASE_ID.items()}
DISC_POINT_INT2STR = {v: k for k, v in SPINAL_DISC_ID.items()}
DISC_DISEASE_INT2STR = {v: k for k, v in SPINAL_DISC_DISEASE_ID.items()}


def extract_point_feature(feature_map: torch.Tensor, coord, size):
    """
    :param feature_map: (batch_size, channels, height, width)
    :param coord: (batch_size, n_points, 2), width在前，height在后
    :param size: coord对应的图片大小
    :return: (batch_size, n_points, channels)
    """
    ratio = torch.tensor([feature_map.shape[-2] / size[0], feature_map.shape[-1] / size[1]], device=coord.device)
    coord = (coord * ratio).round().long()
    image_indices = torch.arange(coord.shape[0]).unsqueeze(1).expand(-1, coord.shape[1]).flatten()
    width_indices = coord[:, :, 0].flatten()
    height_indices = coord[:, :, 1].flatten()
    feature = feature_map.permute(0, 2, 3, 1)
    feature = feature[image_indices, width_indices, height_indices]
    feature = feature.reshape(*coord.shape[:2], -1)
    return feature


class DiseaseModel(torch.nn.Module):
    def __init__(self, kp_model: KeyPointModel, k_nearest, sagittal_size, transverse_size, agg_method='max',
                 num_vertebra_points=len(SPINAL_VERTEBRA_ID), num_vertebra_diseases=len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_points=len(SPINAL_DISC_ID), num_disc_diseases=len(SPINAL_DISC_DISEASE_ID),
                 threshold=0, dropout=0, loss=DisLoss()):
        super().__init__()
        self.kp_model = kp_model
        self.k_nearest = k_nearest
        # height, width
        self.sagittal_size = sagittal_size
        self.transverse_size = transverse_size

        assert agg_method in {'avg', 'max'}
        self.agg_method = agg_method

        self.num_vertebra_points = num_vertebra_points
        self.num_disc_points = num_disc_points

        if agg_method == 'max':
            adaptive_pooling = torch.nn.AdaptiveMaxPool1d(self.out_channels)
        else:
            adaptive_pooling = torch.nn.AdaptiveAvgPool1d(self.out_channels)

        self.adaptive_pooling = torch.nn.Sequential(
            adaptive_pooling,
            torch.nn.LayerNorm(self.out_channels)
        )

        for name in ['sagittal', 'transverse']:
            bottle_neck = torch.nn.Sequential(
                torch.nn.Dropout(dropout, inplace=True),
                torch.nn.Linear(self.out_channels, self.out_channels),
            )
            self.add_module(name + '_bottle_neck', bottle_neck)

        self.layer_norm = torch.nn.LayerNorm(self.out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        self.vertebra_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(self.kp_model.out_channels, num_vertebra_diseases)
        )

        self.disc_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(self.kp_model.out_channels, num_disc_diseases)
        )

        self.threshold = threshold
        self.loss = loss

    @property
    def out_channels(self):
        return self.kp_model.out_channels

    @property
    def kp_loss(self):
        return self.kp_model.loss

    def _cal_transverse_feature(self, transverse: torch.Tensor) -> torch.Tensor:
        """

        :param transverse: (batch_size, n_points, k_nearest, 1, height, width)
        :return: (batch_size, n_points, channels)
        """
        feature = self.kp_model.cal_backbone(transverse.flatten(end_dim=2))
        feature = feature.reshape(*transverse.shape[:3], feature.shape[-3], -1)
        if self.agg_method == 'max':
            feature = feature.max(dim=-1)[0]
            feature = feature.max(dim=-2)[0]
        else:
            feature = feature.mean(dim=-1)
            feature = feature.mean(dim=-2)
        feature = self.adaptive_pooling(feature)
        return feature

    def _cal_final_feature(self, sagittal_feature, transverse_feature):
        sagittal_feature = self.sagittal_bottle_neck(sagittal_feature)
        transverse_feature = self.transverse_bottle_neck(transverse_feature)
        if self.agg_method == 'max':
            final_feature = torch.max(torch.stack([sagittal_feature, transverse_feature], dim=-1), dim=-1)[0]
        else:
            final_feature = sagittal_feature + transverse_feature
        final_feature = self.layer_norm(final_feature)
        final_feature = self.relu(final_feature)
        return final_feature

    def _cal_disease_score(self, feature: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param feature: (batch_num, num_vertebra_points+num_vertebra_points, channels)
        :return: (batch_num, num_vertebra_points, num_vertebra_diseases),
                 (batch_num, num_disc_points, num_disc_diseases)
        """
        vertebra_feature = feature[:, :self.num_vertebra_points]
        disc_feature = feature[:, self.num_vertebra_points:]

        vertebra_score = self.vertebra_head(vertebra_feature)
        disc_score = self.disc_head(disc_feature)
        return vertebra_score, disc_score

    def _inference_sagittal(self, study: Study):
        """

        :param study:
        :return: (1, n_points, 2), (1, n_points, out_channels)
        """
        kp_frame = study.t2_sagittal_middle_frame
        sagittal = tf.resize(kp_frame.image, self.sagittal_size)
        sagittal = tf.to_tensor(sagittal)

        kp_score, sagittal_feature_map = self.kp_model.cal_scores(sagittal.unsqueeze(0))
        kp_heatmap = kp_score.sigmoid_()
        pixel_coord = self.kp_model.spinal_model(kp_heatmap)

        sagittal_feature = extract_point_feature(sagittal_feature_map, pixel_coord, sagittal.shape[-2:])
        return pixel_coord, sagittal_feature

    def _gen_annotation(self, study: Study, coords: torch.Tensor, vertebra_scores, disc_scores) -> dict:
        """

        :param study:
        :param coords: Nx2
        :param vertebra_scores: Vx1
        :param disc_scores: Dx1
        :return:
        """
        point = []
        for i, (coord, score) in enumerate(zip(coords[:self.num_vertebra_points], vertebra_scores)):
            vertebra = torch.where(score > self.threshold)[0].cpu().numpy().tolist()
            if len(vertebra) == 0:
                vertebra = 'v1,'
            else:
                vertebra = [VERTEBRA_DISEASE_INT2STR[j] for j in vertebra]
                vertebra = ','.join(vertebra) + ','
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': VERTEBRA_POINT_INT2STR[i],
                    'vertebra': vertebra
                }
            })
        for i, (coord, score) in enumerate(zip(coords[self.num_vertebra_points:], disc_scores)):
            disc = torch.where(score > self.threshold)[0].cpu().numpy().tolist()
            if len(disc) == 0:
                disc = 'v1,'
            else:
                disc = [DISC_DISEASE_INT2STR[j] for j in disc]
                disc = ','.join(disc) + ','
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': DISC_POINT_INT2STR[i],
                    'disc': disc
                }
            })
        annotation = {
            'studyUid': study.study_uid,
            'data': [
                {
                    'instanceUid': study.t2_sagittal_middle_frame.instance_uid,
                    'seriesUid': study.t2_sagittal_middle_frame.series_uid,
                    'annotation': [
                        {
                            'point': point,
                        }
                    ]
                }
            ]
        }
        return annotation

    def forward(self, sagittal_images, transverse_images, vertebra_labels, disc_labels, distmaps=None):
        """

        :param sagittal_images: (batch_size, 1, height, width)
        :param transverse_images: (batch_size, k_nearest, 1, height, width)
        :param vertebra_labels: (batch_size, num_vertebra_points, 2 + num_vertebra_diseases)
        :param disc_labels: (batch_size, num_disc_points, 2 + num_disc_disease)
        :param distmaps:
        :return:
        """
        kp_score, feature_map = self.kp_model.cal_scores(sagittal_images)

        coord = torch.cat([vertebra_labels[:, :, :2], disc_labels[:, :, :2]], dim=1)

        sagittal_feature = extract_point_feature(feature_map, coord, sagittal_images.shape)

        transverse_feature = self._cal_transverse_feature(transverse_images)
        final_feature = self._cal_final_feature(sagittal_feature, transverse_feature)
        vertebra_score, disc_score = self._cal_disease_score(final_feature)
        if distmaps is None or self.kp_loss is None or self.loss is None:
            return kp_score, vertebra_score, disc_score
        else:
            vertebra_mask = (vertebra_labels[:, :, :2] != PADDING_VALUE).any(dim=-1)
            disc_mask = (disc_labels[:, :, :2] != PADDING_VALUE).any(dim=-1)
            coord_mask = torch.cat([vertebra_mask, disc_mask], dim=1)
            kp_loss = self.kp_loss(kp_score, distmaps, coord_mask)
            vertebra_loss = self.loss(vertebra_score, vertebra_labels[:, :, 2:], vertebra_mask)
            disc_loss = self.loss(disc_score, disc_labels[:, :, 2:], disc_mask)
            return torch.stack([kp_loss, vertebra_loss, disc_loss], dim=0),

    def inference(self, study: Study) -> dict:
        pixel_coord, sagittal_feature = self._inference_sagittal(study)

        transverse = study.t2_transverse_k_nearest(pixel_coord[0].cpu(), self.k_nearest, self.transverse_size)
        transverse = transverse.unsqueeze(0)

        transverse_feature = self._cal_transverse_feature(transverse)
        final_feature = self._cal_final_feature(sagittal_feature, transverse_feature)
        vertebra_score, disc_score = self._cal_disease_score(final_feature)
        return self._gen_annotation(study, pixel_coord[0], vertebra_score[0], disc_score[0])
