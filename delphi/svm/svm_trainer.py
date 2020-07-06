from typing import Any

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model import Model
from delphi.model_trainer import TrainingStyle, DataRequirement
from delphi.svm.feature_cache import FeatureCache
from delphi.svm.svm_model import SVMModel
from delphi.svm.svm_trainer_base import SVMTrainerBase, C_VALUES, GAMMA_VALUES


class SVMTrainer(SVMTrainerBase):

    def __init__(self, context: ModelTrainerContext, model_dir: str, feature_extractor: str, cache: FeatureCache,
                 probability: bool, linear_only: bool):
        super().__init__(context, model_dir, feature_extractor, cache, probability)
        self._param_grid = [{'C': C_VALUES, 'kernel': ['linear']}]

        if not linear_only:
            self._param_grid.append({'C': C_VALUES, 'gamma': GAMMA_VALUES, 'kernel': ['rbf']})

    @property
    def data_requirement(self) -> DataRequirement:
        return DataRequirement.MASTER_ONLY

    @property
    def training_style(self) -> TrainingStyle:
        return TrainingStyle.MASTER_ONLY

    def train_model(self, train_dir: str) -> Model:
        version = self.get_new_version()
        model, _, _ = self.get_best_model(train_dir, self._param_grid)
        return SVMModel(model, version, self.feature_provider, self.probability)

    def message_internal(self, request: Any) -> Any:
        pass
