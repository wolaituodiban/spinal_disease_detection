import torch
import torchvision.transforms.functional as tf
from PIL import Image


def visilize_annotation(image: Image.Image, *annotation: torch.Tensor, _range=10) -> Image.Image:
    """
    关于annotation的结构请参考read_annotation
    :param image:
    :param annotation:
    :param _range:
    :return:
    """
    image = tf.to_tensor(image)
    for _ in annotation:
        for point in _:
            # 注意，image和tensor存在转置关系
            image[0, int(point[1]-_range):int(point[1]+_range), int(point[0]-_range):int(point[0]+_range)] = 0
    return tf.to_pil_image(image)


def visilize_label(image: torch.Tensor, *label: torch.Tensor, max_dist=8) -> Image.Image:
    """
    关于label的结构请参考gen_label
    :param image:
    :param label:
    :param max_dist:
    :return:
    """
    image = image.clone().detach()
    print()
    for _ in label:
        image[(_ < max_dist).sum(dim=-1).bool().unsqueeze(0)] = 0
    return tf.to_pil_image(image)
