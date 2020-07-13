from abc import ABCMeta, abstractmethod
from typing import Iterator

from delphi.proto.learning_module_pb2 import InferObject


class Retriever(metaclass=ABCMeta):

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_objects(self) -> Iterator[InferObject]:
        pass
