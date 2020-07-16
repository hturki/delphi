import queue
from abc import ABCMeta, abstractmethod
from typing import List

from delphi.model import Model


class ReexaminationStrategy(metaclass=ABCMeta):

    @abstractmethod
    def reexamine(self, model: Model, queues: List[queue.PriorityQueue]):
        pass
