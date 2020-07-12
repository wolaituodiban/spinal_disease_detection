from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ..data_utils import gen_mask
from ..structure import Study


class DisDataSet(Dataset):
    def __init__(self,
                 studies: Dict[Any, Study],
                 annotations: Dict[Any, Tuple[torch.Tensor, torch.Tensor]],
                 prob_rotate: float,
                 max_angel: float,
                 num_rep: int,
                 sagittal_size: Tuple[int, int],
                 transverse_size: Tuple[int, int],
                 k_nearest: int,
                 max_dist: int,
                 sagittal_shift: int):
        self.studies = studies
        self.annotations = []
        for k, annotation in annotations.items():
            study_uid, series_uid, instance_uid = k
            if study_uid not in self.studies:
                continue
            study = self.studies[study_uid]
            if series_uid in study and instance_uid in study[series_uid].instance_uids:
                self.annotations.append((k, annotation))

        self.prob_rotate = prob_rotate
        self.max_angel = max_angel
        self.num_rep = num_rep
        self.sagittal_size = sagittal_size
        self.transverse_size = transverse_size
        self.k_nearest = k_nearest
        self.max_dist = max_dist
        self.sagittal_shift = sagittal_shift

    def __len__(self):
        return len(self.annotations) * self.num_rep

    def __getitem__(self, item) -> (Study, Any, (torch.Tensor, torch.Tensor)):
        item = item % len(self.annotations)
        key, (v_annotation, d_annotation) = self.annotations[item]
        return self.studies[key[0]], v_annotation, d_annotation

    def collate_fn(self, data: List[Tuple[Study, torch.Tensor, torch.Tensor]]) -> (Tuple[torch.Tensor], Tuple[None]):
        sagittal_images, transverse_images, vertebra_labels, disc_labels, distmaps = [], [], [], [], []
        v_masks, d_masks, t_masks = [], [], []
        for study, v_anno, d_anno in data:
            # 先构造mask
            v_mask = gen_mask(v_anno)
            d_mask = gen_mask(d_anno)
            v_masks.append(v_mask)
            d_masks.append(d_mask)

            # 然后构造数据
            transverse_image, sagittal_image, distmap, pixel_coord, t_mask = study.transform(
                v_coords=v_anno[:, :2], d_coords=d_anno[:, :2], transverse_size=self.transverse_size,
                sagittal_size=self.sagittal_size, k_nearest=self.k_nearest, max_dist=self.max_dist,
                prob_rotate=self.prob_rotate, max_angel=self.max_angel, sagittal_shift=self.sagittal_shift
            )
            sagittal_images.append(sagittal_image)
            distmaps.append(distmap)
            t_masks.append(t_mask)
            transverse_images.append(transverse_image)

            # 最后构造标签
            v_label = torch.cat([pixel_coord[:v_anno.shape[0]], v_anno[:, 2:]], dim=-1)
            d_label = torch.cat([pixel_coord[v_anno.shape[0]:], d_anno[:, 2:]], dim=-1)
            vertebra_labels.append(v_label)
            disc_labels.append(d_label)

        sagittal_images = torch.stack(sagittal_images, dim=0)
        distmaps = torch.stack(distmaps, dim=0)
        transverse_images = torch.stack(transverse_images, dim=0)
        vertebra_labels = torch.stack(vertebra_labels, dim=0)
        disc_labels = torch.stack(disc_labels, dim=0)
        v_masks = torch.stack(v_masks, dim=0)
        d_masks = torch.stack(d_masks, dim=0)
        t_masks = torch.stack(t_masks, dim=0)

        data = (sagittal_images, transverse_images, distmaps, vertebra_labels, disc_labels, v_masks, d_masks, t_masks)
        label = (None, )
        return data, label

    def gen_sampler(self):
        v_annos, d_annos = [], []
        for key, (v_anno, d_anno) in self.annotations:
            v_annos.append(v_anno[:, -1])
            d_annos.append(d_anno[:, -1])
        v_annos = torch.stack(v_annos, dim=0)
        d_annos = torch.stack(d_annos, dim=0)

        v_count = torch.unique(v_annos, return_counts=True)[1]
        d_count = torch.unique(d_annos, return_counts=True)[1]

        v_weights = torch.true_divide(1, torch.cumprod(v_count[v_annos], dim=-1)[:, -1])
        d_weights = torch.true_divide(1, torch.cumprod(d_count[d_annos], dim=-1)[:, -1])

        weights = v_weights * d_weights
        return WeightedRandomSampler(weights=weights, num_samples=len(self), replacement=True)


class DisDataLoader(DataLoader):
    def __init__(self, studies, annotations, batch_size, sagittal_size, transverse_size, k_nearest, prob_rotate=False,
                 max_angel=0, max_dist=8, sagittal_shift=0, num_workers=0,  num_rep=1, pin_memory=False,
                 sampling_strategy=None):
        assert sampling_strategy in {'balance', None}
        dataset = DisDataSet(studies=studies, annotations=annotations, sagittal_size=sagittal_size,
                             transverse_size=transverse_size, k_nearest=k_nearest, prob_rotate=prob_rotate,
                             max_angel=max_angel, num_rep=num_rep, max_dist=max_dist, sagittal_shift=sagittal_shift)
        if sampling_strategy == 'balance':
            sampler = dataset.gen_sampler()
            super().__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                             pin_memory=pin_memory, collate_fn=dataset.collate_fn)
        else:
            super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory, collate_fn=dataset.collate_fn)
