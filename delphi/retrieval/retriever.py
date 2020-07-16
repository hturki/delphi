from abc import ABCMeta, abstractmethod
from typing import Iterator, Iterable

from delphi.proto.learning_module_pb2 import DelphiObject


class Retriever(metaclass=ABCMeta):

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_objects(self) -> Iterator[DelphiObject]:
        pass

    @abstractmethod
    def get_object(self, object_id: str, attributes: Iterable[str]) -> DelphiObject:
        pass
