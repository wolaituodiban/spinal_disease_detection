from typing import Any, Dict, Tuple

import torch
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from ..data_utils import resize, random_rotate, gen_label, PADDING_VALUE


class PosDataSet:
    def __init__(self,
                 images: Dict[Any, Image.Image],
                 spacings: Dict[Any, torch.Tensor],
                 annotation: Dict[Any, torch.Tensor],
                 size: Tuple[int, int],
                 rotate,
                 num_rep):
        self.images = []
        self.spacings = []
        self.coords = []
        self.masks = []
        self.rotate = rotate
        self.num_rep = num_rep

        for k, v in tqdm(annotation.items(), ascii=True):
            coord = torch.cat([_[:, :2] for _ in v], dim=0)
            image, spacing = images[k], spacings[k]
            image, spacing, coord = resize(size, image, spacing, coord)
            mask = (coord != PADDING_VALUE).any(dim=1)
            mask = mask.reshape(mask.size(0), 1, 1)
            self.images.append(image)
            self.spacings.append(spacing)
            self.coords.append(coord)
            self.masks.append(mask)

    def __len__(self):
        return len(self.masks) * self.num_rep

    def __getitem__(self, i):
        i = i % len(self.masks)

        image, spacing, coord, mask = self.images[i], self.spacings[i], self.coords[i], self.masks[i]
        if self.rotate:
            image, coord = random_rotate(image, coord)
        image = tf.to_tensor(image)
        label = gen_label(image, spacing, coord)
        return (image, label, mask), (label, mask)


class PosDataLoader(DataLoader):
    def __init__(self, images, spacings, annotation, size, batch_size, num_workers=0, rotate=False, num_rep=1):
        dataset = PosDataSet(images, spacings, annotation, size, rotate, num_rep)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         pin_memory=True)
