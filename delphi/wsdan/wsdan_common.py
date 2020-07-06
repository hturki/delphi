import random

import torch
import torch.nn as nn
from logzero import logger

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = nn.functional.interpolate(atten_map, size=(imgH, imgW), mode='bilinear',
                                                  align_corners=True) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            if len(nonzero_indices) > 0:
                height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
                height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
                width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
                width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            else:
                logger.warn('No non-zero indices found - taking whole image')
                height_min = 0
                height_max = imgH
                width_min = 0
                width_max = imgW

            crop_images.append(nn.functional.interpolate(
                images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                size=(imgH, imgW), mode='bilinear', align_corners=True))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(
                nn.functional.interpolate(atten_map, size=(imgH, imgW), mode='bilinear', align_corners=True) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError(
            'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)
