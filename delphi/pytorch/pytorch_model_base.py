import multiprocessing as mp
from abc import abstractmethod
from pathlib import Path
from typing import List, Callable, Iterable, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from delphi.model import Model
from delphi.object_provider import ObjectProvider
from delphi.provider_and_result import ProviderAndResult
from delphi.pytorch.pytorch_trainer_base import get_test_transforms


def preprocess(request: ObjectProvider) -> Tuple[ObjectProvider, torch.Tensor]:
    return request, get_test_transforms()(request.get_content())


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

    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ProviderAndResult]:
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

    def _process_batch(self, batch: List[Tuple[ObjectProvider, torch.Tensor]]) -> Iterable[ProviderAndResult]:
        tensors = torch.stack([f[1] for f in batch]).to(self._device, non_blocking=True)
        predictions = self.get_predictions(tensors)
        for i in range(len(batch)):
            score = predictions[i]
            yield ProviderAndResult(batch[i][0], '1' if score >= 0.5 else '0', score, self.version)
