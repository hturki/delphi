from abc import ABCMeta, abstractmethod
from typing import Mapping

# Must be picklable
class AttributeProvider(metaclass=ABCMeta):

    @abstractmethod
    def get(self) -> Mapping[str, bytes]:
        pass
