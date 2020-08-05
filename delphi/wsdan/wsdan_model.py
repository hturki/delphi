import io
import uuid
from typing import List, Any, Optional

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from delphi.pytorch.pytorch_model_base import PytorchModelBase
from delphi.wsdan.wsdan_common import batch_augment, STD, MEAN

TEST_BATCH_SIZE = 16

WSDAN_TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class WSDANModel(PytorchModelBase):

    def __init__(self, model: nn.Module, version: int, tb_writer: Optional[SummaryWriter], epoch: int,
                 optimizer_dict: Any, feature_center: torch.Tensor):
        super().__init__(WSDAN_TEST_TRANSFORMS, TEST_BATCH_SIZE, version)

        self._model = model
        self._version = version
        self._tb_writer = tb_writer

        # These are just kept in case we want to resume training from this model. They're not actually necessary
        # for inference
        self._epoch = epoch
        self._optimizer_dict = optimizer_dict
        self._feature_center = feature_center

    def get_predictions(self, input: torch.Tensor) -> List[float]:
        with torch.no_grad():

            # WS-DAN
            y_pred_raw, _, attention_maps = self._model(input)

            # Augmentation with crop_mask
            crop_image = batch_augment(input, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _ = self._model(crop_image)
            y_pred = (y_pred_raw + y_pred_crop) / 2.

            predicted = torch.softmax(y_pred, dim=1)[:, 1].tolist()

            if self._tb_writer is not None:
                for i in range(len(predicted)):
                    if predicted[i] > 0.5:
                        # reshape attention maps
                        image_attention_map = nn.functional.interpolate(attention_maps[i].unsqueeze(0),
                                                                        size=(input.size(2), input.size(3)),
                                                                        mode='bilinear', align_corners=True)
                        image_attention_map = torch.sqrt(image_attention_map.cpu() / image_attention_map.max().item())

                        # get heat attention maps
                        heat_attention_maps = self._generate_heatmap(image_attention_map)

                        # raw_image, heat_attention, raw_attention
                        raw_image = input[i].cpu() * STD + MEAN
                        heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
                        raw_attention_image = raw_image * image_attention_map

                        object_id = str(uuid.uuid4())
                        self._tb_writer.add_image('%s raw' % object_id, raw_image.squeeze())
                        self._tb_writer.add_image('%s raw attention' % object_id, raw_attention_image.squeeze())
                        self._tb_writer.add_image('%s heat attention' % object_id, heat_attention_image.squeeze())

            return predicted

    def get_bytes(self) -> bytes:
        bytes = io.BytesIO()
        torch.save({
            'epoch': self._epoch,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer_dict,
            'feature_center': self._feature_center
        }, bytes)
        bytes.seek(0)

        return bytes.getvalue()

    @staticmethod
    def _generate_heatmap(attention_maps: Any) -> torch.Tensor:
        heat_attention_maps = []
        heat_attention_maps.append(attention_maps[:, 0, ...])  # R
        heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                                   (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
        heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
        return torch.stack(heat_attention_maps, dim=1)
