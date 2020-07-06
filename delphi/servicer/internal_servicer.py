from typing import Iterator

import grpc
from google.protobuf.any_pb2 import Any
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import BytesValue, Int32Value

from delphi.coordinator import Coordinator
from delphi.proto.internal_pb2 import GetExamplesRequest, ExampleMetadata, GetExampleRequest, StageModelRequest, \
    TestResult, \
    InternalMessage
from delphi.proto.internal_pb2_grpc import InternalServiceServicer
from delphi.utils import log_exceptions_and_abort


class InternalServicer(InternalServiceServicer):

    def __init__(self, coordinator: Coordinator):
        self._coordinator = coordinator

    @log_exceptions_and_abort
    def GetExamples(self, request: GetExamplesRequest, context: grpc.ServicerContext) -> Iterator[ExampleMetadata]:
        return self._coordinator.get_examples(request.exampleSet, request.nodeIndex)

    @log_exceptions_and_abort
    def GetExample(self, request: GetExampleRequest, context: grpc.ServicerContext) -> BytesValue:
        with open(self._coordinator.get_example(request.exampleSet, request.label, request.key), 'rb') as f:
            return BytesValue(value=f.read())

    @log_exceptions_and_abort
    def Train(self, request: Int32Value, context: grpc.ServicerContext) -> Empty:
        self._coordinator.train(request.value)
        return Empty()

    @log_exceptions_and_abort
    def StageModel(self, request: StageModelRequest, context: grpc.ServicerContext) -> Empty:
        self._coordinator.stage_model(request.version, request.trainerIndex, request.content)
        return Empty()

    @log_exceptions_and_abort
    def ValidateTestResults(self, request: Int32Value, context: grpc.ServicerContext) -> Empty:
        self._coordinator.validate_test_results(request.value)
        return Empty()

    @log_exceptions_and_abort
    def SubmitTestResults(self, request: Iterator[TestResult], context: grpc.ServicerContext) -> Empty:
        self._coordinator.submit_test_results(request)
        return Empty()

    @log_exceptions_and_abort
    def PromoteModel(self, request: Int32Value, context: grpc.ServicerContext) -> Empty:
        self._coordinator.promote_model(request.value)
        return Empty()

    @log_exceptions_and_abort
    def DiscardModel(self, request: Int32Value, context: grpc.ServicerContext) -> Empty:
        self._coordinator.promote_model(request.value)
        return Empty()

    @log_exceptions_and_abort
    def MessageInternal(self, request: InternalMessage, context: grpc.ServicerContext) -> Any:
        return self._coordinator.message_internal(request.trainerIndex, request.message)
