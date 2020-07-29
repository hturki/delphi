import multiprocessing as mp
from abc import abstractmethod
from pathlib import Path
from typing import List, Callable, Iterable, Tuple, Any

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from delphi.model import Model
from delphi.proto.learning_module_pb2 import InferResult, DelphiObject
from delphi.pytorch.pytorch_trainer_base import get_test_transforms


def preprocess(request: DelphiObject) -> Tuple[str, Any, torch.Tensor]:
    return request.objectId, request.attributes, get_test_transforms()(request.content)


class PytorchModelBase(Model):
    test_transforms: transforms.Compose

    def __init__(self, test_transforms: transforms.Compose, batch_size: int, version: int, pool: mp.Pool):
        self._batch_size = batch_size
        self._test_transforms = test_transforms
        self._version = version
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._pool = pool

    @abstractmethod
    def get_predictions(self, input: torch.Tensor) -> List[float]:
        pass

    @property
    def version(self) -> int:
        return self._version

    def infer(self, requests: Iterable[DelphiObject]) -> Iterable[InferResult]:
        batch = []
        items = self._pool.imap_unordered(preprocess, requests)

        for item in items:
            batch.append(item)
            if len(batch) == self._batch_size:
                yield from self._process_batch(batch)
                batch = []

        if len(batch) > 0:
            yield from self._process_batch(batch)

    def infer_dir(self, directory: Path, callback_fn: Callable[[int, float], None]) -> None:
        dataset = datasets.ImageFolder(str(directory), transform=self._test_transforms)
        data_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=mp.cpu_count())

        with torch.no_grad():
            for _, (input, target) in enumerate(data_loader):
                predictions = self.get_predictions(input.to(self._device, non_blocking=True))

                for i in range(len(predictions)):
                    callback_fn(target[i].tolist(), predictions[i])

    @property
    def scores_are_probabilities(self) -> bool:
        return True

    def _process_batch(self, batch: List[Tuple[str, Any, torch.Tensor]]) -> Iterable[InferResult]:
        tensors = torch.stack([f[2] for f in batch]).to(self._device, non_blocking=True)
        predictions = self.get_predictions(tensors)
        for i in range(len(batch)):
            score = predictions[i]
            yield InferResult(objectId=batch[i][0], attributes=batch[i][1], label='1' if score >= 0.5 else '0',
                              score=score, modelVersion=self.version)
