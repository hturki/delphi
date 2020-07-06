from abc import abstractmethod
from typing import List

from delphi.context.context_base import ContextBase
from delphi.model_trainer import ModelTrainer


class DataManagerContext(ContextBase):

    @property
    @abstractmethod
    def data_dir(self) -> str:
        pass

    @abstractmethod
    def get_active_trainers(self) -> List[ModelTrainer]:
        pass

    @abstractmethod
    def new_examples_callback(self, new_positives: int, new_negatives: int) -> None:
        pass


