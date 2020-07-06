from typing import Dict, Optional

from delphi.condition.model_condition import ModelCondition
from delphi.model_trainer import ModelTrainer
from delphi.proto.learning_module_pb2 import ModelStatistics


class TestAucCondition(ModelCondition):

    def __init__(self, threshold: float, trainer: ModelTrainer):
        super().__init__()
        self._threshold = threshold
        self._trainer = trainer

    def is_satisfied(self, example_counts: Dict[str, int], last_statistics: Optional[ModelStatistics]) -> bool:
        return last_statistics.auc > self._threshold

    @property
    def trainer(self) -> ModelTrainer:
        return self._trainer
