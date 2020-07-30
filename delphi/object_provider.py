from abc import ABCMeta, abstractmethod
from typing import Mapping

# Must be picklable
class ObjectProvider(metaclass=ABCMeta):

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def get_content(self) -> bytes:
        pass

    @abstractmethod
    def get_attributes(self) -> Mapping[str, bytes]:
        pass