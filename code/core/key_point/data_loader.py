import math
import random
from typing import Any, Dict

import torch
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from ..data_utils import resize, rotate, gen_label, PADDING_VALUE


class KeyPointDataSet:
    def __init__(self,
                 images: Dict[Any, Image.Image],
                 spacings: Dict[Any, torch.Tensor],
                 annotation: Dict[Any, torch.Tensor],
                 random_resize: bool,
                 prob_rotate: float,
                 max_angel: float,
                 num_rep: int,
                 prob_reverse: float):
        self.images = []
        self.spacings = []
        self.coords = []
        self.masks = []
        self.should_resize = random_resize
        self.prob_ratate = prob_rotate
        self.max_angel = max_angel
        self.num_rep = num_rep
        self.prob_reverse = prob_reverse

        self.max_width = -math.inf
        self.max_height = -math.inf
        self.min_width = math.inf
        self.min_height = math.inf

        for k, v in tqdm(annotation.items(), ascii=True):
            coord = torch.cat([_[:, :2] for _ in v], dim=0)
            image, spacing = images[k], spacings[k]
            mask = (coord != PADDING_VALUE).any(dim=1)
            mask = mask.reshape(mask.size(0), 1, 1)
            self.images.append(image)
            self.spacings.append(spacing)
            self.coords.append(coord)
            self.masks.append(mask)
            width, height = image.size
            self.max_height = max(self.max_height, height)
            self.max_width = max(self.max_width, width)
            self.min_height = min(self.min_height, height)
            self.min_width = min(self.min_width, width)

    def __len__(self):
        return len(self.masks) * self.num_rep

    def __getitem__(self, i):
        i = i % len(self.masks)
        return self.images[i], self.spacings[i], self.coords[i], self.masks[i]

    def collate_fn(self, data):
        if self.should_resize:
            size = (random.randint(self.min_width, self.max_width), random.randint(self.min_height, self.max_height))
        else:
            size = None
        images, labels, masks = [], [], []
        for image, spacing, coord, mask in data:
            if size is not None:
                image, spacing, coord = resize(size, image, spacing, coord)

            if self.max_angel > 0 and random.random() <= self.prob_ratate:
                angel = random.randint(-self.max_angel, self.max_angel)
                image, coord = rotate(image, coord, angel)
                image = tf.to_tensor(image)
                label = gen_label(image, spacing, coord, angel=-angel)
            else:
                image = tf.to_tensor(image)
                label = gen_label(image, spacing, coord)

            if self.prob_reverse > 0 and random.random() <= self.prob_reverse:
                image = 1 - image

            images.append(image)
            labels.append(label)
            masks.append(mask)
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        masks = torch.stack(masks, dim=0)
        return (images, labels, masks), (labels, masks)


class KeyPointDataLoader(DataLoader):
    def __init__(self, images, spacings, annotation, batch_size, num_workers=0, random_resize=False, prob_rotate=False,
                 max_angel=0, num_rep=1, prob_reverse=0):
        dataset = KeyPointDataSet(images, spacings, annotation, random_resize=random_resize, prob_rotate=prob_rotate,
                                  max_angel=max_angel, num_rep=num_rep, prob_reverse=prob_reverse)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         pin_memory=True, collate_fn=dataset.collate_fn)
