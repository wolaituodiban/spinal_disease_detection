import sys

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

sys.path.append('../nn_tools/')
from nn_tools import torch_utils

from .data_loader import KeyPointDataLoader
from .loss import KeyPointBCELossV2, NullLoss
from .evaluation import KeyPointAcc
from .model import *
from .spinal_model import SpinalModel
from ..data_utils import read_dcms, get_spacing, read_annotation, SPINAL_DISC_ID, SPINAL_VERTEBRA_ID


if __name__ == '__main__':
    train_images, train_metainfos = read_dcms('data/lumbar_train150/')
    valid_images, valid_metainfos = read_dcms('data/train/')

    train_spacings = get_spacing(train_metainfos)
    valid_spacings = get_spacing(valid_metainfos)

    train_annotation = read_annotation('data/lumbar_train150_annotation.json')
    valid_annotation = read_annotation('data/lumbar_train51_annotation.json')
    valid_annotation = {k: v for k, v in valid_annotation.items() if k in valid_images}

    train_pk_dataloader = KeyPointDataLoader(
        train_images, train_spacings, train_annotation, batch_size=8, num_workers=4,
        prob_rotate=1, max_angel=180, num_rep=20, prob_reverse=0, size=[512, 512],
        pin_memory=False
    )
    valid_pk_dataloader = KeyPointDataLoader(
        valid_images, valid_spacings, valid_annotation, batch_size=1, num_workers=4,
        num_rep=20, size=[512, 512], pin_memory=False
    )

    backbone = resnet_fpn_backbone('resnet50', True)
    spinal_model = SpinalModel(train_images, train_annotation,
                               num_candidates=128, num_selected_templates=8,
                               max_translation=0.05, scale_range=(0.9, 1.1), max_angel=10)
    kp_model = KeyPointModelV2(backbone, len(SPINAL_VERTEBRA_ID), len(SPINAL_DISC_ID),
                               pixel_mean=torch.tensor(0.5), pixel_std=torch.tensor(1), dropout=0,
                               loss=KeyPointBCELossV2(lamb=1), spinal_model=spinal_model, loss_scaler=100,
                               num_cascades=2
                               ).cuda(0)
    temp = torch.load('../models/size512_rotate1_180_AdamW_1e-5.kp_model', map_location='cuda:0')
    kp_model.backbone.load_state_dict(temp.backbone.state_dict())
    kp_model.fc.load_state_dict(temp.fc.state_dict())

    optimizer = torch.optim.AdamW(kp_model.parameters(), lr=1e-5)
    max_step = 50*len(train_pk_dataloader)
    result = torch_utils.fit(
        kp_model,
        train_pk_dataloader,
        valid_pk_dataloader,
        optimizer,
        max_step,
        NullLoss(),
        [KeyPointAcc(6)],
        is_higher_better=True,
        evaluate_per_steps=len(train_pk_dataloader),
        checkpoint_dir='models',
    )
