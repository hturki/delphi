from abc import ABCMeta, abstractmethod
from typing import Iterator

from delphi.model import Model
from delphi.proto.learning_module_pb2 import InferResult


class Selector(metaclass=ABCMeta):

    @abstractmethod
    def add_result(self, result: InferResult) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass

    @abstractmethod
    def get_results(self) -> Iterator[InferResult]:
        pass

    @abstractmethod
    def new_model(self, model: Model) -> None:
        pass
