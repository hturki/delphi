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
from delphi.result_provider import ResultProvider
from delphi.utils import log_exceptions


def preprocess(request: ObjectProvider) -> Tuple[ObjectProvider, torch.Tensor]:
    get_semaphore().acquire()

    return request, get_test_transforms()(request.content)


class PytorchModelBase(Model):
    test_transforms: transforms.Compose

    def __init__(self, test_transforms: transforms.Compose, batch_size: int, version: int):
        self._batch_size = batch_size
        self._test_transforms = test_transforms
        self._version = version
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def get_predictions(self, input: torch.Tensor) -> List[float]:
        pass

    @property
    def version(self) -> int:
        return self._version

    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ResultProvider]:
        semaphore = mp.Semaphore(256)  # Make sure that the load function doesn't overload the consumer
        batch = []

        with mp.Pool(min(16, mp.cpu_count()), initializer=init_worker,
                     initargs=(self._test_transforms, semaphore)) as pool:
            items = pool.imap_unordered(preprocess, requests)

            for item in items:
                semaphore.release()
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

    def _process_batch(self, batch: List[Tuple[ObjectProvider, torch.Tensor]]) -> Iterable[ResultProvider]:
        tensors = torch.stack([f[1] for f in batch]).to(self._device, non_blocking=True)
        predictions = self.get_predictions(tensors)
        for i in range(len(batch)):
            score = predictions[i]
            yield ResultProvider(batch[i][0].id, '1' if score >= 0.5 else '0', score, self.version,
                                 batch[i][0].attributes, batch[i][0].gt)


_test_transforms: transforms.Compose
_semaphore: mp.Semaphore


def get_test_transforms() -> transforms.Compose:
    global _test_transforms
    return _test_transforms


def get_semaphore() -> mp.Semaphore:
    global _semaphore
    return _semaphore


@log_exceptions
def init_worker(test_transforms: transforms.Compose, semaphore: mp.Semaphore):
    global _test_transforms
    _test_transforms = test_transforms

    global _semaphore
    _semaphore = semaphore
