import uuid
from pathlib import Path
from typing import Iterator

import grpc
from google.protobuf import json_format
from google.protobuf.empty_pb2 import Empty
from logzero import logger
from opendiamond.client.search import DiamondSearch, FilterSpec, Blob
from opendiamond.scope import ScopeCookie

from delphi.condition.examples_per_label_condition import ExamplesPerLabelCondition
from delphi.condition.test_auc_condition import TestAucCondition
from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.learning_module_stub import LearningModuleStub
from delphi.mpncov.mpncov_trainer import MPNCovTrainer
from delphi.proto.search_pb2 import SearchRequest, SearchId, RetrainPolicyConfig, SVMMode, SVMConfig, Dataset, \
    SelectorConfig, ReexaminationStrategyConfig
from delphi.proto.search_pb2_grpc import SearchServiceServicer
from delphi.retrain.absolute_threshold_policy import AbsoluteThresholdPolicy
from delphi.retrain.percentage_threshold_policy import PercentageThresholdPolicy
from delphi.retrain.retrain_policy import RetrainPolicy
from delphi.retrieval.diamond_retriever import DiamondRetriever
from delphi.retrieval.retriever import Retriever
from delphi.search import Search
from delphi.search_manager import SearchManager
from delphi.selection.full_reexamination_strategy import FullReexaminationStrategy
from delphi.selection.no_reexamination_strategy import NoReexaminationStrategy
from delphi.selection.reexamination_strategy import ReexaminationStrategy
from delphi.selection.selector import Selector
from delphi.selection.top_reexamination_strategy import TopReexaminationStrategy
from delphi.selection.topk_selector import TopKSelector
from delphi.svm.distributed_svm_trainer import DistributedSVMTrainer
from delphi.svm.ensemble_svm_trainer import EnsembleSVMTrainer
from delphi.svm.feature_cache import FeatureCache
from delphi.svm.svm_trainer import SVMTrainer
from delphi.svm.svm_trainer_base import SVMTrainerBase
from delphi.utils import log_exceptions_and_abort
from delphi.wsdan.wsdan_trainer import WSDANTrainer


class SearchServicer(SearchServiceServicer):

    def __init__(self, manager: SearchManager, root_dir: Path, model_dir: Path, feature_cache: FeatureCache, port: int):
        self._manager = manager
        self._root_dir = root_dir
        self._model_dir = model_dir
        self._feature_cache = feature_cache
        self._port = port

    @log_exceptions_and_abort
    def Start(self, request: Iterator[SearchRequest], context: grpc.ServicerContext) -> SearchId:
        config = next(request).config
        retrain_policy = self._get_retrain_policy(config.retrainPolicy)

        search_id = str(uuid.uuid4())
        key = SearchId(value=search_id)
        search = Search(key, config.nodeIndex, [LearningModuleStub(node) for node in config.nodes], retrain_policy,
                        config.onlyUseBetterModels, self._root_dir / search_id, self._port)

        trainers = []
        for i in range(len(config.trainStrategy)):
            if config.trainStrategy[i].HasField('examplesPerLabel'):
                condition_builder = lambda x: ExamplesPerLabelCondition(config.trainStrategy[i].examplesPerLabel.count,
                                                                        x)
                model = config.trainStrategy[i].examplesPerLabel.model
            elif config.trainStrategy[i].HasField('testAuc'):
                condition_builder = lambda x: TestAucCondition(config.trainStrategy[i].testAuc.threshold, x)
                model = config.trainStrategy[i].testAuc.model
            else:
                raise NotImplementedError(
                    'unknown condition: {}'.format(json_format.MessageToJson(config.trainStrategy[i])))

            if model.HasField('svm'):
                trainer = self._get_svm_trainer(search, key, i, model.svm)
            elif model.HasField('fastMPNCOV'):
                trainer = MPNCovTrainer(search, model.fastMPNCOV.distributed, model.fastMPNCOV.freeze, self._model_dir)
            elif model.HasField('wsdan'):
                trainer = WSDANTrainer(search, model.wsdan.distributed, model.wsdan.visualize, model.wsdan.freeze)
            else:
                raise NotImplementedError('unknown model: {}'.format(json_format.MessageToJson(model)))

            trainers.append(condition_builder(trainer))

        search.trainers = trainers
        logger.info(
            'Starting search with id {} and parameters:\n{}'.format(search_id, json_format.MessageToJson(config)))

        search.start(self._get_retriever(config.dataset), (x.example for x in request))

        self._manager.set_search(key, search)
        return key

    @log_exceptions_and_abort
    def Stop(self, request: SearchId, context: grpc.ServicerContext) -> Empty:
        self._manager.remove_search(request)
        return Empty()

    def _get_retrain_policy(self, retrain_policy: RetrainPolicyConfig) -> RetrainPolicy:
        if retrain_policy.HasField('absolute'):
            return AbsoluteThresholdPolicy(retrain_policy.absolute.threshold, retrain_policy.absolute.onlyPositives)
        elif retrain_policy.HasField('percentage'):
            return PercentageThresholdPolicy(retrain_policy.percentage.threshold,
                                             retrain_policy.percentage.onlyPositives)
        else:
            raise NotImplementedError('unknown retrain policy: {}'.format(json_format.MessageToJson(retrain_policy)))

    def _get_selector(self, selector: SelectorConfig) -> Selector:
        if selector.HasField('topk'):
            return TopKSelector(selector.topk.k, selector.topk.batchSize,
                                self._get_reexamination_strategy(selector.topk.reexaminationStrategy))
        else:
            raise NotImplementedError('unknown selector: {}'.format(json_format.MessageToJson(selector)))

    def _get_reexamination_strategy(self, reexamination_strategy: ReexaminationStrategyConfig) -> ReexaminationStrategy:
        if reexamination_strategy.HasField('none'):
            return NoReexaminationStrategy()
        elif reexamination_strategy.HasField('top'):
            return TopReexaminationStrategy(reexamination_strategy.top.k)
        elif reexamination_strategy.HasField('full'):
            return FullReexaminationStrategy()
        else:
            raise NotImplementedError(
                'unknown reexamination strategy: {}'.format(json_format.MessageToJson(reexamination_strategy)))

    def _get_svm_trainer(self, context: ModelTrainerContext, search_id: SearchId, trainer_index: int,
                         config: SVMConfig) -> SVMTrainerBase:
        feature_extractor = config.featureExtractor
        probability = config.probability
        linear_only = config.linearOnly

        if config.mode is SVMMode.MASTER_ONLY:
            return SVMTrainer(context, self._model_dir, feature_extractor, self._feature_cache, probability,
                              linear_only)
        elif config.mode is SVMMode.DISTRIBUTED:
            return DistributedSVMTrainer(context, self._model_dir, feature_extractor, self._feature_cache, probability,
                                         linear_only, search_id, trainer_index)
        elif config.mode is SVMMode.ENSEMBLE:
            if not config.probability:
                raise NotImplementedError('Probability must be enabled when using ensemble SVM trainer')

            return EnsembleSVMTrainer(context, self._model_dir, feature_extractor, self._feature_cache, linear_only,
                                      search_id, trainer_index)
        else:
            raise NotImplementedError('unknown svm mode: {}'.format(config.mode))

    def _get_retriever(self, dataset: Dataset) -> Retriever:
        if dataset.HasField('diamond'):
            diamond_search = DiamondSearch([ScopeCookie.parse(x) for x in dataset.diamond.cookies],
                                           [FilterSpec(x.name, x.code, x.arguments, Blob(x.blob), x.dependencies,
                                                       x.minScore, x.maxScore) for x in dataset.diamond.filters],
                                           dataset.diamond.attributes)
            return DiamondRetriever(diamond_search)
        else:
            raise NotImplementedError('unknown dataset: {}'.format(json_format.MessageToJson(dataset)))
