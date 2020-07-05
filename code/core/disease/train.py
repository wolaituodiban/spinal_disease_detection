import sys
import time

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .data_loader import DisDataLoader
from .evaluation import Evaluator
from .model import DiseaseModel
from ..data_utils import SPINAL_DISC_ID, SPINAL_VERTEBRA_ID
from ..key_point import SpinalModel, KeyPointModelV2, KeyPointBCELossV2, NullLoss
from ..structure import construct_studies

sys.path.append('../nn_tools/')
from nn_tools import torch_utils


if __name__ == '__main__':
    start_time = time.time()
    train_studies, train_annotation, train_counter = construct_studies(
        '../data/lumbar_train150', '../data/lumbar_train150_annotation.json', multiprocessing=True)
    valid_studies, valid_annotation, valid_counter = construct_studies(
        '../data/train/', '../data/lumbar_train51_annotation.json', multiprocessing=True)

    train_dataloader = DisDataLoader(train_studies, train_annotation, batch_size=8, sagittal_size=[512, 512],
                                     transverse_size=[256, 256], k_nearest=1, num_workers=4, num_rep=20,
                                     prob_rotate=1, max_angel=180)

    train_images = {}
    for study_uid, study in train_studies.items():
        frame = study.t2_sagittal_middle_frame
        train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image

    backbone = resnet_fpn_backbone('resnet50', True)
    spinal_model = SpinalModel(train_images, train_annotation,
                               num_candidates=128, num_selected_templates=8,
                               max_translation=0.05, scale_range=(0.9, 1.1), max_angel=10)
    kp_model = KeyPointModelV2(backbone, len(SPINAL_VERTEBRA_ID), len(SPINAL_DISC_ID),
                               pixel_mean=0.5, pixel_std=1,
                               loss=KeyPointBCELossV2(lamb=1), spinal_model=spinal_model, loss_scaler=100,
                               num_cascades=2)
    kp_model.load_state_dict(torch.load('../models/pretrained.kp_model_v2'))
    dis_model = DiseaseModel(
        kp_model, sagittal_size=(512, 512), use_kp_loss=False, share_backbone=False
    ).cuda()

    valid_evaluator = Evaluator(
        dis_model, valid_studies, '../data/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
        metric='macro precision'
    )

    optimizer = torch.optim.AdamW(dis_model.parameters(), lr=1e-5)
    max_step = 5 * len(train_dataloader)
    fit_result = torch_utils.fit(
        dis_model,
        train_data=train_dataloader,
        valid_data=None,
        optimizer=optimizer,
        max_step=max_step,
        loss=NullLoss(),
        metrics=[valid_evaluator.metric],
        is_higher_better=True,
        evaluate_per_steps=len(train_dataloader),
        evaluate_fn=valid_evaluator,
    )

    dis_model.kp_model = None
    torch.save(dis_model.cpu().state_dict(), '../models/pretrained.dis_model')
    print('task completed, {} seconds used'.format(time.time() - start_time))
