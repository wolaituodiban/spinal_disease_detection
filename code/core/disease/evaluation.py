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
    def __init__(self, module: DiseaseModel, studies: Dict[str, Study], annotation_path: str, metric='macro f1',
                 max_dist=8, epsilon=1e-5, num_rep=1):
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
                coord = point['coord']
                if 'disc' in point['tag']:
                    temp[identification] = {
                        'coord': coord,
                        'disease': point['tag']['disc'],
                    }
                else:
                    temp[identification] = {
                        'coord': coord,
                        'disease': point['tag']['vertebra'],
                    }

            self.annotations.append({
                'studyUid': study_uid,
                'seriesUid': series_uid,
                'instanceUid': instance_uid,
                'annotation': temp
            })
        self.annotations *= num_rep
        self.metric = metric
        self.max_dist = max_dist
        self.epsilon = epsilon

    def __call__(self, *args, **kwargs):
        self.module.eval()
        kp_tp = {}
        tp = {}
        fp = {}
        fn = {}

        for annotation in tqdm(self.annotations, ascii=True):
            study = self.studies[annotation['studyUid']]
            pixel_spacing = study.t2_sagittal_middle_frame.pixel_spacing
            pred = self.module(study, to_dict=True)
            for point in pred['data'][0]['annotation'][0]['data']['point']:
                identification = point['tag']['identification']
                if identification not in annotation['annotation']:
                    continue
                gt_point = annotation['annotation'][identification]
                gt_coord = gt_point['coord']
                gt_disease = gt_point['disease'][:2]
                if gt_disease == '':
                    gt_disease = 'v1'

                if 'disc' in point['tag']:
                    disease_type = 'disc'
                else:
                    disease_type = 'vertebra'
                coord = point['coord']
                disease = point['tag'][disease_type]

                disease_type = disease_type + ' ' + gt_disease
                if disease_type not in tp:
                    tp[disease_type] = self.epsilon
                if disease_type not in fp:
                    fp[disease_type] = self.epsilon
                if disease_type not in kp_tp:
                    kp_tp[disease_type] = self.epsilon
                if disease_type not in fn:
                    fn[disease_type] = self.epsilon

                if distance(coord, gt_coord, pixel_spacing) < self.max_dist:
                    kp_tp[disease_type] += 1
                    if disease == gt_disease:
                        tp[disease_type] += 1
                    else:
                        fp[disease_type] += 1
                else:
                    fn[disease_type] += 1
        kp_accuracy = {k: kp_tp[k] / (kp_tp[k] + fn[k]) for k in kp_tp}
        d_precision = {k: tp[k] / (tp[k] + fp[k]) for k in tp}
        d_recall = {k: tp[k] / (tp[k] + fn[k]) for k in tp}
        d_f1 = {k: 2 * d_precision[k] * d_recall[k] / (d_precision[k] + d_recall[k]) for k in d_precision}

        output = []
        for k, v in d_precision.items():
            output.append((k + ' precision', v))
        output = sorted(output, key=lambda x: x[0])

        sum_tp = sum(tp.values())
        sum_fp = sum(fp.values())
        sum_fn = sum(fn.values())
        micro_precision = sum_tp / (sum_tp + sum_fp)
        micro_recall = sum_tp / (sum_tp + sum_fn)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        macro_f1 = sum(d_f1.values()) / len(d_f1)
        macro_precision = sum(d_precision.values()) / len(d_precision)
        avg_acc = sum(kp_accuracy.values()) / len(kp_accuracy)
        output = [('macro f1', macro_f1), ('micro f1', micro_f1), ('avg key point acc', avg_acc),
                  ('macro precision', macro_precision)] + output

        i = 0
        while i < len(output) and output[i][0] != self.metric:
            i += 1
        if i < len(output):
            output = [output[i]] + output[:i] + output[i+1:]
        return output
