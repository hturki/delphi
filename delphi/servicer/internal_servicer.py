from typing import Iterable

import grpc
from google.protobuf.any_pb2 import Any
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import BytesValue

from delphi.proto.internal_pb2 import GetExamplesRequest, ExampleMetadata, GetExampleRequest, StageModelRequest, \
    InternalMessage, TrainModelRequest, ValidateTestResultsRequest, SubmitTestRequest, PromoteModelRequest, \
    DiscardModelRequest
from delphi.proto.internal_pb2_grpc import InternalServiceServicer
from delphi.search_manager import SearchManager
from delphi.utils import log_exceptions_and_abort


class InternalServicer(InternalServiceServicer):

    def __init__(self, manager: SearchManager):
        self._manager = manager

    @log_exceptions_and_abort
    def GetExamples(self, request: GetExamplesRequest, context: grpc.ServicerContext) -> Iterable[ExampleMetadata]:
        return self._manager.get_search(request.searchId).get_examples(request.exampleSet, request.nodeIndex)

    @log_exceptions_and_abort
    def GetExample(self, request: GetExampleRequest, context: grpc.ServicerContext) -> BytesValue:
        example_path = self._manager.get_search(request.searchId).get_example(request.exampleSet, request.label,
                                                                              request.key)
        with example_path.open('rb') as f:
            return BytesValue(value=f.read())

    @log_exceptions_and_abort
    def TrainModel(self, request: TrainModelRequest, context: grpc.ServicerContext) -> Empty:
        self._manager.get_search(request.searchId).train_model(request.trainerIndex)
        return Empty()

    @log_exceptions_and_abort
    def StageModel(self, request: StageModelRequest, context: grpc.ServicerContext) -> Empty:
        self._manager.get_search(request.searchId).stage_model(request.version, request.trainerIndex, request.content)
        return Empty()

    @log_exceptions_and_abort
    def ValidateTestResults(self, request: ValidateTestResultsRequest, context: grpc.ServicerContext) -> Empty:
        self._manager.get_search(request.searchId).validate_test_results(request.version)
        return Empty()

    @log_exceptions_and_abort
    def SubmitTestResults(self, request: Iterable[SubmitTestRequest], context: grpc.ServicerContext) -> Empty:
        search_id = next(request).searchId
        self._manager.get_search(search_id).submit_test_results(x.result for x in request)
        return Empty()

    @log_exceptions_and_abort
    def PromoteModel(self, request: PromoteModelRequest, context: grpc.ServicerContext) -> Empty:
        self._manager.get_search(request.searchId).promote_model(request.version)
        return Empty()

    @log_exceptions_and_abort
    def DiscardModel(self, request: DiscardModelRequest, context: grpc.ServicerContext) -> Empty:
        self._manager.get_search(request.searchId).promote_model(request.version)
        return Empty()

    @log_exceptions_and_abort
    def MessageInternal(self, request: InternalMessage, context: grpc.ServicerContext) -> Any:
        return self._manager.get_search(request.searchId).message_internal(request.trainerIndex, request.message)
