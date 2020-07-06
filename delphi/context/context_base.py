from abc import ABCMeta, abstractmethod
from typing import List

from delphi.learning_module_stub import LearningModuleStub


class ContextBase(metaclass=ABCMeta):

    @property
    @abstractmethod
    def node_index(self) -> int:
        pass

    @property
    @abstractmethod
    def nodes(self) -> List[LearningModuleStub]:
        pass
