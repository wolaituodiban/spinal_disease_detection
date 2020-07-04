import sys
import time

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .data_loader import KeyPointDataLoader
from .loss import KeyPointBCELossV2, NullLoss
from .evaluation import KeyPointAcc
from .model import KeyPointModel
from .spinal_model import SpinalModel
from ..data_utils import read_dcms, get_spacing, read_annotation, SPINAL_DISC_ID, SPINAL_VERTEBRA_ID

sys.path.append('../nn_tools/')
from nn_tools import torch_utils

if __name__ == '__main__':
    start_time = time.time()
    train_images, train_metainfos = read_dcms('data/lumbar_train150/')
    valid_images, valid_metainfos = read_dcms('data/train/')

    train_spacings = get_spacing(train_metainfos)
    valid_spacings = get_spacing(valid_metainfos)

    train_annotation = read_annotation('data/lumbar_train150_annotation.json')
    valid_annotation = read_annotation('data/lumbar_train51_annotation.json')
    valid_annotation = {k: v for k, v in valid_annotation.items() if k in valid_images}

    train_dataloader = KeyPointDataLoader(
        train_images, train_spacings, train_annotation, batch_size=8, num_workers=3,
        prob_rotate=1, max_angel=180, num_rep=20, size=[512, 512],
        pin_memory=True
    )
    valid_dataloader = KeyPointDataLoader(
        valid_images, valid_spacings, valid_annotation, batch_size=1, num_workers=5,
        num_rep=20, size=[512, 512], pin_memory=True
    )

    # stage one
    backbone = resnet_fpn_backbone('resnet50', True)
    spinal_model = SpinalModel(train_images, train_annotation,
                               num_candidates=128, num_selected_templates=8,
                               max_translation=0.05, scale_range=(0.9, 1.1), max_angel=10)
    kp_model = KeyPointModel(backbone, len(SPINAL_VERTEBRA_ID), len(SPINAL_DISC_ID),
                             pixel_mean=0.5, pixel_std=1,
                             loss=KeyPointBCELossV2(lamb=1), spinal_model=spinal_model).cuda()

    optimizer = torch.optim.AdamW(kp_model.parameters(), lr=1e-5)
    max_step = 30*len(train_dataloader)
    result = torch_utils.fit(
        kp_model,
        train_dataloader,
        valid_dataloader,
        optimizer,
        max_step,
        NullLoss(),
        [KeyPointAcc(6)],
        is_higher_better=True,
        evaluate_per_steps=len(train_dataloader),
        # checkpoint_dir='models',
    )
    torch.save(kp_model.cpu().state_dict(), 'models/pretrained.kp_model')
    print('task completed, {} seconds used'.format(time.time() - start_time))
