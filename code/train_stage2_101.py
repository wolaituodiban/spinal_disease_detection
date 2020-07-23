import sys
import time

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from code.core.disease import DisDataLoader, DiseaseModelBase, Evaluator
from code.core.key_point import KeyPointModelV2, NullLoss, SpinalModel, KeyPointBCELossV2, CascadeLossV2
from code.core.structure import construct_studies

sys.path.append('../nn_tools/')
from nn_tools import torch_utils


if __name__ == '__main__':
    start_time = time.time()
    train_studies, train_annotation, train_counter = construct_studies(
        '../data/lumbar_train150', '../data/lumbar_train150_annotation.json', multiprocessing=True)
    valid_studies, valid_annotation, valid_counter = construct_studies(
        '../data/lumbar_train51/', '../data/lumbar_train51_annotation.json', multiprocessing=True)

    # 设定模型参数
    train_images = {}
    for study_uid, study in train_studies.items():
        frame = study.t2_sagittal_middle_frame
        train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image

    backbone = resnet_fpn_backbone('resnet101', True)
    spinal_model = SpinalModel(train_images, train_annotation,
                               num_candidates=128, num_selected_templates=8,
                               max_translation=0.05, scale_range=(0.9, 1.1), max_angel=10)
    kp_model = KeyPointModelV2(backbone, pixel_mean=0.5, pixel_std=1,
                               loss=KeyPointBCELossV2(lamb=1), spinal_model=spinal_model,
                               cascade_loss=CascadeLossV2(1), loss_scaler=100, num_cascades=3)
    kp_model.load_state_dict(torch.load('../models/pretrained101.kp_model'), strict=False)

    dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))
    dis_model.cuda(0)
    print(dis_model)

    # 设定训练参数
    train_dataloader = DisDataLoader(
        train_studies, train_annotation, batch_size=8, num_workers=3, num_rep=20, prob_rotate=1, max_angel=180,
        sagittal_size=dis_model.sagittal_size, transverse_size=dis_model.sagittal_size, k_nearest=0, max_dist=6,
        sagittal_shift=1, pin_memory=False
    )

    valid_evaluator = Evaluator(
        dis_model, valid_studies, '../data/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
        metric='key point recall'
    )

    step_per_batch = len(train_dataloader)
    optimizer = torch.optim.AdamW(dis_model.parameters(), lr=5e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=step_per_batch, T_mult=2)
    max_step = 50 * step_per_batch
    fit_result = torch_utils.fit(
        dis_model,
        train_data=train_dataloader,
        valid_data=None,
        optimizer=optimizer,
        scheduler=scheduler,
        max_step=max_step,
        loss=NullLoss(),
        metrics=[valid_evaluator.metric],
        is_higher_better=True,
        evaluate_per_steps=step_per_batch,
        evaluate_fn=valid_evaluator,
    )

    torch.save(dis_model.backbone.cpu().state_dict(), '../models/pretrained101.kp_model_v2')
    print('task completed, {} seconds used'.format(time.time() - start_time))
