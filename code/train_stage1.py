import sys
import time

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .core.disease import DisDataLoader, DiseaseModelBase, Evaluator
from .core.key_point import KeyPointModel, NullLoss, SpinalModel, KeyPointBCELossV2
from .core.structure import construct_studies

sys.path.append('../nn_tools/')
from nn_tools import torch_utils


if __name__ == '__main__':
    start_time = time.time()
    train_studies, train_annotation, train_counter = construct_studies(
        'data/lumbar_train150', 'data/lumbar_train150_annotation.json', multiprocessing=True)
    valid_studies, valid_annotation, valid_counter = construct_studies(
        'data/train/', 'data/lumbar_train51_annotation.json', multiprocessing=True)

    # 设定模型参数
    train_images = {}
    for study_uid, study in train_studies.items():
        frame = study.t2_sagittal_middle_frame
        train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image

    backbone = resnet_fpn_backbone('resnet50', True)
    spinal_model = SpinalModel(train_images, train_annotation,
                               num_candidates=128, num_selected_templates=8,
                               max_translation=0.05, scale_range=(0.9, 1.1), max_angel=10)
    kp_model = KeyPointModel(backbone, pixel_mean=0.5, pixel_std=1,
                             loss=KeyPointBCELossV2(lamb=1), spinal_model=spinal_model).cuda()
    dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))
    dis_model.cuda()
    print(dis_model)

    # 设定训练参数
    train_dataloader = DisDataLoader(
        train_studies, train_annotation, batch_size=8, num_workers=3, num_rep=20, prob_rotate=1, max_angel=180,
        sagittal_size=dis_model.sagittal_size, transverse_size=dis_model.sagittal_size, k_nearest=0, max_dist=6,
        sagittal_shift=1, pin_memory=True
    )

    valid_evaluator = Evaluator(
        dis_model, valid_studies, 'data/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
        metric='key point recall'
    )

    step_per_batch = len(train_dataloader)
    optimizer = torch.optim.AdamW(dis_model.parameters(), lr=1e-5)
    max_step = 30 * step_per_batch
    fit_result = torch_utils.fit(
        dis_model,
        train_data=train_dataloader,
        valid_data=None,
        optimizer=optimizer,
        max_step=max_step,
        loss=NullLoss(),
        metrics=[valid_evaluator.metric],
        is_higher_better=True,
        evaluate_per_steps=step_per_batch,
        evaluate_fn=valid_evaluator,
    )

    torch.save(dis_model.backbone.cpu().state_dict(), 'models/pretrained.kp_model')
    print('task completed, {} seconds used'.format(time.time() - start_time))
