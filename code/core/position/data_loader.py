from typing import Any, Dict, Tuple

import torch
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from ..data_utils import resize, gen_label, PADDING_VALUE


class PosDataSet:
    def __init__(self,
                 images: Dict[Any, Image.Image],
                 spacings: Dict[Any, torch.Tensor],
                 annotation: Dict[Any, torch.Tensor],
                 size: Tuple[int, int]):
        self.data = []
        self.label = []
        self.mask = []

        for k, v in tqdm(annotation.items(), ascii=True):
            coord = torch.cat([_[:, :2] for _ in v], dim=0)
            image, spacing = images[k], spacings[k]
            image, spacing, coord = resize(size, image, spacing, coord)
            image = tf.to_tensor(image)
            label = gen_label(image, spacing, coord)
            mask = (coord != PADDING_VALUE).any(dim=1)
            mask = mask.reshape(mask.size(0), 1, 1)
            self.data.append(image)
            self.label.append(label)
            self.mask.append(mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (self.data[i], self.label[i], self.mask[i]), (self.label[i], self.mask[i])


class PosDataLoader(DataLoader):
    def __init__(self, images, spacings, annotation, size, batch_size, num_workers=0):
        dataset = PosDataSet(images, spacings, annotation, size)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         pin_memory=True)
