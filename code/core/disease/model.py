import torch
import torchvision.transforms.functional as tf
from ..structure import Study
from ..key_point import KeyPointModel
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_VERTEBRA_DISEASE_ID, SPINAL_DISC_ID, SPINAL_DISC_DISEASE_ID


class DiseaseModel(torch.nn.Module):
    def __init__(self, kp_model: KeyPointModel, k_nearest, sagittal_size, transverse_size, agg_method='max',
                 num_vertebra_points=len(SPINAL_VERTEBRA_ID), num_vertebra_diseases=len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_points=len(SPINAL_DISC_ID), num_disc_diseases=len(SPINAL_DISC_DISEASE_ID)):
        super().__init__()
        self.kp_model = kp_model
        self.k_nearest = k_nearest
        # height, width
        self.sagittal_size = sagittal_size
        self.transverse_size = transverse_size

        assert agg_method in {'sum', 'max'}
        self.agg_method = agg_method

        self.vertebra_head = torch.nn.Linear(self.kp_model.out_channels, num_vertebra_diseases)
        self.disc_head = torch.nn.Linear(self.kp_model.out_channels, num_disc_diseases)

        self.num_vertebra_points = num_vertebra_points
        self.num_disc_points = num_disc_points

    def get_key_points(self, study: Study):
        kp_frame = study.t2_sagittal_middle_frame

        height_ratio = self.sagittal_size[0] / kp_frame.size[1]
        width_ratio = self.sagittal_size[1] / kp_frame.size[0]

        image = tf.resize(kp_frame.image, self.sagittal_size)
        image = tf.to_tensor(image)

        pixel_coord = self.kp_model(image.unsqueeze(0))[0][0]

        ratio = torch.tensor([width_ratio, height_ratio], device=pixel_coord.device)
        pixel_coord = pixel_coord.float() / ratio
        return pixel_coord

    def inference(self, study: Study):
        pixel_coord = self.get_key_points(study)
        human_coord = study.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord.cpu())

        nearest_transverse = []
        for series in study.t2_transverse.k_nearest(human_coord, self.k_nearest):
            for dicom in series:
                image = tf.to_tensor(tf.resize(dicom.image, self.transverse_size))
                nearest_transverse.append(image)
        nearest_transverse = torch.stack(nearest_transverse, dim=0)

        feature = self.kp_model.cal_feature_maps(nearest_transverse)['pool']
        feature = feature.reshape(-1, self.k_nearest, *feature.shape[1:])
        feature = feature.flatten(start_dim=2)
        if self.agg_method == 'max':
            feature = feature.max(dim=1)[0]
            feature = feature.max(dim=-1)[0]
        else:
            feature = feature.sum(dim=1)
            feature = feature.sum(dim=-1)
        return feature
