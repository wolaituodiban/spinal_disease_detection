from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from ..structure import DICOM, Study


class DisDataSet(Dataset):
    def __init__(self,
                 studies: Dict[Any, Study],
                 annotations: Dict[Any, Tuple[torch.Tensor, torch.Tensor]],
                 prob_rotate: float,
                 max_angel: float,
                 num_rep: int,
                 sagittal_size: Tuple[int, int],
                 transverse_size: Tuple[int, int],
                 k_nearest: int):
        self.studies = studies
        self.annotations = []
        for k, annotation in annotations.items():
            study_uid, series_uid, instance_uid = k
            if study_uid not in self.studies:
                continue
            study = self.studies[study_uid]
            if series_uid in study and instance_uid in study[series_uid].instance_uids:
                self.annotations.append((k, annotation))

        self.prob_rotate = prob_rotate,
        self.max_angel = max_angel
        self.num_rep = num_rep
        self.sagittal_size = sagittal_size
        self.transverse_size = transverse_size
        self.k_nearest = k_nearest

    def __len__(self):
        return len(self.annotations) * self.num_rep

    def __getitem__(self, item):
        item = item % len(self)
        key, annotation = self.annotations[item]
        return self.studies[key[0]], key, annotation

    def collate_fn(self, data) -> (Tuple[torch.Tensor], Tuple[None]):
        sagittal_images, transverse_images, vertebra_labels, disc_labels, distmaps = [], [], [], [], []
        for study, key, annotation in data:
            vertebra_labels.append(annotation[0])
            disc_labels.append(annotation[1])

            pixel_coord = torch.cat([_[:, :2] for _ in annotation], dim=0)
            dicom: DICOM = study[key[1]][key[2]]
            sagittal_image, sagittal_distmap = dicom.transform(
                pixel_coord, self.sagittal_size, self.prob_rotate, self.max_angel)
            sagittal_images.append(sagittal_image)
            distmaps.append(sagittal_distmap)

            transverse_image = study.t2_transverse_k_nearest(
                pixel_coord, k=self.k_nearest, size=self.transverse_size,
                prob_rotate=self.prob_rotate, max_angel=self.max_angel
            )
            # padding
            if transverse_image is None:
                transverse_image = torch.zeros(pixel_coord.shape[0], self.k_nearest, 1, *self.transverse_size)
            transverse_images.append(transverse_image)

        sagittal_images = torch.stack(sagittal_images, dim=0)
        distmaps = torch.stack(distmaps, dim=0)
        transverse_images = torch.stack(transverse_images, dim=0)
        vertebra_labels = torch.stack(vertebra_labels)
        disc_labels = torch.stack(disc_labels)
        return (sagittal_images, transverse_images, vertebra_labels, disc_labels, distmaps), (None, )


class DisDataLoader(DataLoader):
    def __init__(self, studies, annotations, batch_size, sagittal_size, transverse_size, k_nearest,
                 num_workers=0, prob_rotate=False, max_angel=0, num_rep=1, pin_memory=True):
        dataset = DisDataSet(studies=studies, annotations=annotations, sagittal_size=sagittal_size,
                             transverse_size=transverse_size, k_nearest=k_nearest, prob_rotate=prob_rotate,
                             max_angel=max_angel, num_rep=num_rep)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         pin_memory=pin_memory, collate_fn=dataset.collate_fn)
