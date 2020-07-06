import io
import multiprocessing as mp
from typing import List, Any

import torch
import torchvision.transforms as transforms
from torch import nn

from delphi.pytorch.pytorch_model_base import PytorchModelBase

TEST_BATCH_SIZE = 48

MPNCOV_TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MPNCovModel(PytorchModelBase):

    def __init__(self, model: nn.Module, version: int, epoch: int, optimizer_dict: Any, pool: mp.Pool):
        super().__init__(MPNCOV_TEST_TRANSFORMS, TEST_BATCH_SIZE, version, pool)

        self._model = model
        self._version = version

        # These are just kept in case we want to resume training from this model. They're not actually necessary
        # for inference
        self._epoch = epoch
        self._optimizer_dict = optimizer_dict

    def get_predictions(self, input: torch.Tensor) -> List[float]:
        with torch.no_grad():
            output = self._model(input)
            return torch.softmax(output, dim=1)[:, 1].tolist()

    def get_bytes(self) -> bytes:
        bytes = io.BytesIO()
        torch.save({
            'epoch': self._epoch,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer_dict,
        }, bytes)
        bytes.seek(0)

        return bytes.getvalue()
