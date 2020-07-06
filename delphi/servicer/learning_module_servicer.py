from typing import Iterator

import grpc
from google.protobuf.empty_pb2 import Empty

from delphi.coordinator import Coordinator
from delphi.proto.learning_module_pb2 import InferRequest, InferResult, LabeledExample, ModelStatistics, \
    ImportModelRequest, \
    ModelArchive
from delphi.proto.learning_module_pb2_grpc import LearningModuleServiceServicer
from delphi.utils import log_exceptions_and_abort


class LearningModuleServicer(LearningModuleServiceServicer):

    def __init__(self, coordinator: Coordinator):
        self._coordinator = coordinator

    @log_exceptions_and_abort
    def Infer(self, request: Iterator[InferRequest], context: grpc.ServicerContext) -> Iterator[InferResult]:
        yield from self._coordinator.infer(request)

    @log_exceptions_and_abort
    def AddLabeledExamples(self, request: Iterator[LabeledExample], context: grpc.ServicerContext) -> Empty:
        self._coordinator.add_labeled_examples(request)
        return Empty()

    @log_exceptions_and_abort
    def GetModelStatistics(self, request: Empty, context: grpc.ServicerContext) -> ModelStatistics:
        return self._coordinator.get_model_statistics()

    @log_exceptions_and_abort
    def ImportModel(self, request: ImportModelRequest, context: grpc.ServicerContext) -> Empty:
        self._coordinator.import_model(request.version, request.content)
        return Empty()

    @log_exceptions_and_abort
    def ExportModel(self, request: Empty, context: grpc.ServicerContext) -> ModelArchive:
        return self._coordinator.export_model()
