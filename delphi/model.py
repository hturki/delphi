from abc import ABCMeta, abstractmethod
from typing import Callable, Iterator

from delphi.proto.learning_module_pb2 import InferResult, InferRequest


class Model(metaclass=ABCMeta):

    @property
    @abstractmethod
    def version(self) -> int:
        pass

    @abstractmethod
    def infer(self, requests: Iterator[InferRequest]) -> Iterator[InferResult]:
        pass

    @abstractmethod
    def infer_dir(self, directory: str, callback_fn: Callable[[int, float], None]) -> None:
        pass

    @abstractmethod
    def get_bytes(self) -> bytes:
        pass

    @property
    @abstractmethod
    def scores_are_probabilities(self) -> bool:
        pass
