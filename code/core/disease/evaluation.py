import json
import math
from typing import Dict

from tqdm import tqdm

from .model import DiseaseModel
from ..structure import Study


def distance(coord0, coord1, pixel_spacing):
    x = (coord0[0] - coord1[0]) * pixel_spacing[0]
    y = (coord0[1] - coord1[1]) * pixel_spacing[1]
    output = math.sqrt(x ** 2 + y ** 2)
    return output


class Evaluator:
    def __init__(self, module: DiseaseModel, studies: Dict[str, Study], annotation_path: str,
                 max_dist=8, epsilon=1e-5):
        self.module = module
        self.studies = studies
        with open(annotation_path, 'r') as file:
            annotations = json.load(file)

        self.annotations = []
        for annotation in annotations:
            study_uid = annotation['studyUid']
            series_uid = annotation['data'][0]['seriesUid']
            instance_uid = annotation['data'][0]['instanceUid']
            temp = {}
            for point in annotation['data'][0]['annotation'][0]['data']['point']:
                identification = point['tag']['identification']
                if 'disc' in point['tag']:
                    disease = point['tag']['disc']
                else:
                    disease = point['tag']['vertebra']
                coord = point['coord']
                temp[identification] = {
                    'coord': coord,
                    'disease': disease,
                }
            self.annotations.append({
                'studyUid': study_uid,
                'seriesUid': series_uid,
                'instanceUid': instance_uid,
                'annotation': temp
            })
        self.max_dist = max_dist
        self.epsilon = epsilon

    def __call__(self, *args, **kwargs):
        self.module.eval()
        kp_tp = 0
        tp = 0
        fp = 0
        fn = 0

        for annotation in tqdm(self.annotations, ascii=True):
            study = self.studies[annotation['studyUid']]
            pixel_spacing = study.t2_sagittal_middle_frame.pixel_spacing
            pred = self.module(study, to_dict=True)
            for point in pred['data'][0]['annotation'][0]['point']:
                identification = point['tag']['identification']
                if identification not in annotation['annotation']:
                    continue
                gt_point = annotation['annotation'][identification]
                gt_coord = gt_point['coord']
                gt_disease = gt_point['disease'].replace(',', '')
                coord = point['coord']
                if distance(coord, gt_coord, pixel_spacing) < self.max_dist:
                    kp_tp += 1
                    if 'disc' in point['tag']:
                        disease = point['tag']['disc']
                    else:
                        disease = point['tag']['vertebra']

                    if disease == gt_disease:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1
        kp_accuracy = kp_tp / (kp_tp + fn)
        d_precision = tp / (tp + fp) + self.epsilon
        d_recall = tp / (tp + fn) + self.epsilon
        d_f1 = d_precision * d_recall / (d_precision + d_recall)
        return [('disease f1', d_f1), ('key point accuracy', kp_accuracy)]
