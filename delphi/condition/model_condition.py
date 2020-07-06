from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

from delphi.model_trainer import ModelTrainer
from delphi.proto.learning_module_pb2 import ModelStatistics


class ModelCondition(metaclass=ABCMeta):

    @abstractmethod
    def is_satisfied(self, example_counts: Dict[str, int], last_statistics: Optional[ModelStatistics]) -> bool:
        pass

    @property
    @abstractmethod
    def trainer(self) -> ModelTrainer:
        pass
