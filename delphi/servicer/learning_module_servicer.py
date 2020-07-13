from typing import Iterator

import grpc
from google.protobuf.empty_pb2 import Empty

from delphi.proto.learning_module_pb2 import InferRequest, InferResult, ModelStatistics, \
    ImportModelRequest, ModelArchive, LabeledExampleRequest
from delphi.proto.learning_module_pb2_grpc import LearningModuleServiceServicer
from delphi.proto.search_pb2 import SearchId
from delphi.search_manager import SearchManager
from delphi.utils import log_exceptions_and_abort


class LearningModuleServicer(LearningModuleServiceServicer):

    def __init__(self, manager: SearchManager):
        self._manager = manager

    @log_exceptions_and_abort
    def GetResults(self, request: SearchId, context: grpc.ServicerContext) -> Iterator[InferResult]:
        yield from self._manager.get_search(request).selector.get_results()

    @log_exceptions_and_abort
    def Infer(self, request: Iterator[InferRequest], context: grpc.ServicerContext) -> Iterator[InferResult]:
        search_id = next(request).searchId
        yield from self._manager.get_search(search_id).infer(x.object for x in request)

    @log_exceptions_and_abort
    def AddLabeledExamples(self, request: Iterator[LabeledExampleRequest], context: grpc.ServicerContext) -> Empty:
        search_id = next(request).searchId
        self._manager.get_search(search_id).add_labeled_examples(x.example for x in request)
        return Empty()

    @log_exceptions_and_abort
    def GetModelStatistics(self, request: SearchId, context: grpc.ServicerContext) -> ModelStatistics:
        return self._manager.get_search(request).get_model_statistics()

    @log_exceptions_and_abort
    def ImportModel(self, request: ImportModelRequest, context: grpc.ServicerContext) -> Empty:
        self._manager.get_search(request.searchId).import_model(request.version, request.content)
        return Empty()

    @log_exceptions_and_abort
    def ExportModel(self, request: Empty, context: grpc.ServicerContext) -> ModelArchive:
        return self._manager.get_search(request.searchId).export_model()
