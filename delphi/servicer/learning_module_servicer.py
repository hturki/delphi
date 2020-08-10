from pathlib import Path
from typing import Iterable

import grpc
from google.protobuf import json_format
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import Int64Value
from logzero import logger
from opendiamond.server.object_ import ATTR_DATA

from delphi.condition.bandwidth_condition import BandwidthCondition
from delphi.condition.examples_per_label_condition import ExamplesPerLabelCondition
from delphi.condition.test_auc_condition import TestAucCondition
from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.learning_module_stub import LearningModuleStub
from delphi.mpncov.mpncov_trainer import MPNCovTrainer
from delphi.object_provider import ObjectProvider
from delphi.proto.learning_module_pb2 import InferRequest, InferResult, ModelStats, \
    ImportModelRequest, ModelArchive, LabeledExampleRequest, SearchId, \
    AddLabeledExampleIdsRequest, LabeledExample, DelphiObject, GetObjectsRequest, SearchStats, SearchInfo, \
    CreateSearchRequest
from delphi.proto.learning_module_pb2 import RetrainPolicyConfig, SVMMode, SVMConfig, Dataset, \
    SelectorConfig, ReexaminationStrategyConfig
from delphi.proto.learning_module_pb2_grpc import LearningModuleServiceServicer
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
from delphi.simple_attribute_provider import SimpleAttributeProvider
from delphi.svm.distributed_svm_trainer import DistributedSVMTrainer
from delphi.svm.ensemble_svm_trainer import EnsembleSVMTrainer
from delphi.svm.feature_cache import FeatureCache
from delphi.svm.svm_trainer import SVMTrainer
from delphi.svm.svm_trainer_base import SVMTrainerBase
from delphi.utils import log_exceptions_and_abort
from delphi.wsdan.wsdan_trainer import WSDANTrainer


class LearningModuleServicer(LearningModuleServiceServicer):

    def __init__(self, manager: SearchManager, root_dir: Path, model_dir: Path, feature_cache: FeatureCache, port: int):
        self._manager = manager
        self._root_dir = root_dir
        self._model_dir = model_dir
        self._feature_cache = feature_cache
        self._port = port

    @log_exceptions_and_abort
    def CreateSearch(self, request: CreateSearchRequest, context: grpc.ServicerContext) -> Empty:
        retrain_policy = self._get_retrain_policy(request.retrainPolicy)

        nodes = [LearningModuleStub(node) for node in request.nodes]
        search = Search(request.searchId, request.nodeIndex, nodes, retrain_policy, request.onlyUseBetterModels,
                        self._root_dir / request.searchId.value, self._port, self._get_retriever(request.dataset),
                        self._get_selector(request.selector), request.hasInitialExamples)

        trainers = []
        for i in range(len(request.trainStrategy)):
            if request.trainStrategy[i].HasField('examplesPerLabel'):
                condition_builder = lambda x: ExamplesPerLabelCondition(request.trainStrategy[i].examplesPerLabel.count,
                                                                        x)
                model = request.trainStrategy[i].examplesPerLabel.model
            elif request.trainStrategy[i].HasField('testAuc'):
                condition_builder = lambda x: TestAucCondition(request.trainStrategy[i].testAuc.threshold, x)
                model = request.trainStrategy[i].testAuc.model
            elif request.trainStrategy[i].HasField('bandwidth'):
                bandwidth_config = request.trainStrategy[i].bandwidth
                condition_builder = lambda x: BandwidthCondition(request.nodeIndex, nodes,
                                                                 bandwidth_config.thresholdMbps,
                                                                 bandwidth_config.refreshSeconds, x)
                model = request.trainStrategy[i].bandwidth.model
            else:
                raise NotImplementedError(
                    'unknown condition: {}'.format(json_format.MessageToJson(request.trainStrategy[i])))

            if model.HasField('svm'):
                trainer = self._get_svm_trainer(search, request.searchId, i, model.svm)
            elif model.HasField('fastMPNCOV'):
                trainer = MPNCovTrainer(search, model.fastMPNCOV.distributed, model.fastMPNCOV.freeze.value,
                                        self._model_dir)
            elif model.HasField('wsdan'):
                trainer = WSDANTrainer(search, model.wsdan.distributed, model.wsdan.visualize, model.wsdan.freeze)
            else:
                raise NotImplementedError('unknown model: {}'.format(json_format.MessageToJson(model)))

            trainers.append(condition_builder(trainer))

        search.trainers = trainers
        self._manager.set_search(request.searchId, search, request.metadata)

        logger.info('Create search with id {} and parameters:\n{}'.format(request.searchId.value,
                                                                          json_format.MessageToJson(request)))
        return Empty()

    @log_exceptions_and_abort
    def StartSearch(self, request: SearchId, context: grpc.ServicerContext) -> Empty:
        logger.info('Starting search with id {}'.format(request.value))
        self._manager.get_search(request).start()
        return Empty()

    @log_exceptions_and_abort
    def StopSearch(self, request: SearchId, context: grpc.ServicerContext) -> Empty:
        logger.info('Stopping search with id {}'.format(request.value))
        search = self._manager.remove_search(request)
        search.stop()
        return Empty()

    @log_exceptions_and_abort
    def GetSearches(self, request: SearchId, context: grpc.ServicerContext) -> Iterable[SearchInfo]:
        for search_id, metadata in self._manager.get_searches():
            yield SearchInfo(searchId=search_id, metadata=metadata)

    @log_exceptions_and_abort
    def GetResults(self, request: SearchId, context: grpc.ServicerContext) -> Iterable[InferResult]:
        while True:
            result = self._manager.get_search(request).selector.get_result()
            if result is None:
                return
            yield InferResult(objectId=result.id, label=result.label, score=result.score,
                              modelVersion=result.model_version, attributes=result.attributes.get())

    @log_exceptions_and_abort
    def GetObjects(self, request: GetObjectsRequest, context: grpc.ServicerContext) -> Iterable[DelphiObject]:
        retriever = self._get_retriever(request.dataset)
        try:
            retriever.start()
            for object_id in request.objectIds:
                yield retriever.get_object(object_id, request.attributes)
        finally:
            retriever.stop()

    @log_exceptions_and_abort
    def Infer(self, request: Iterable[InferRequest], context: grpc.ServicerContext) -> Iterable[InferResult]:
        search_id = next(request).searchId
        for result in self._manager.get_search(search_id).infer(
                ObjectProvider(x.object.objectId, x.object.content, SimpleAttributeProvider(x.object.attributes), False)
                for x in request):
            yield InferResult(objectId=result.id, label=result.label, score=result.score,
                              modelVersion=result.model_version, attributes=result.attributes.get())

    @log_exceptions_and_abort
    def AddLabeledExamples(self, request: Iterable[LabeledExampleRequest], context: grpc.ServicerContext) -> Empty:
        search_id = next(request).searchId
        self._manager.get_search(search_id).add_labeled_examples(x.example for x in request)
        return Empty()

    @log_exceptions_and_abort
    def AddLabeledExampleIds(self, request: AddLabeledExampleIdsRequest, context: grpc.ServicerContext) -> Empty:
        search = self._manager.get_search(request.searchId)
        examples = []
        for object_id in request.examples:
            example = search.retriever.get_object(object_id, [ATTR_DATA])
            examples.append(LabeledExample(label=request.examples[object_id], content=example.content))

        search.add_labeled_examples(examples)
        return Empty()

    @log_exceptions_and_abort
    def GetSearchStats(self, request: SearchId, context: grpc.ServicerContext) -> SearchStats:
        search = self._manager.get_search(request)
        retriever_stats = search.retriever.get_stats()
        selector_stats = search.selector.get_stats()
        passed_objects = selector_stats.passed_objects
        return SearchStats(totalObjects=retriever_stats.total_objects,
                           processedObjects=retriever_stats.dropped_objects + selector_stats.processed_objects,
                           droppedObjects=retriever_stats.dropped_objects + selector_stats.dropped_objects,
                           passedObjects=Int64Value(value=passed_objects) if passed_objects is not None else None,
                           falseNegatives=retriever_stats.false_negatives + selector_stats.false_negatives)

    @log_exceptions_and_abort
    def GetModelStats(self, request: SearchId, context: grpc.ServicerContext) -> ModelStats:
        return self._manager.get_search(request).get_model_stats()

    @log_exceptions_and_abort
    def ImportModel(self, request: ImportModelRequest, context: grpc.ServicerContext) -> Empty:
        self._manager.get_search(request.searchId).import_model(request.version, request.content)
        return Empty()

    @log_exceptions_and_abort
    def ExportModel(self, request: Empty, context: grpc.ServicerContext) -> ModelArchive:
        return self._manager.get_search(request.searchId).export_model()

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
            return DiamondRetriever(dataset.diamond)
        else:
            raise NotImplementedError('unknown dataset: {}'.format(json_format.MessageToJson(dataset)))
