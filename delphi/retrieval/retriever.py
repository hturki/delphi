from abc import ABCMeta, abstractmethod
from typing import Iterable

from delphi.object_provider import ObjectProvider
from delphi.proto.learning_module_pb2 import DelphiObject


class Retriever(metaclass=ABCMeta):

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_objects(self) -> Iterable[ObjectProvider]:
        pass

    @abstractmethod
    def get_object(self, object_id: str, attributes: Iterable[str]) -> DelphiObject:
        pass
