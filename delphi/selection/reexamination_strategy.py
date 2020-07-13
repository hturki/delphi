import queue
from abc import ABCMeta, abstractmethod
from typing import Iterator, List

from delphi.model import Model
from delphi.proto.learning_module_pb2 import InferResult


class ReexaminationStrategy(metaclass=ABCMeta):

    @abstractmethod
    def reexamine(self, model: Model, queues: List[queue.PriorityQueue]):
        pass
