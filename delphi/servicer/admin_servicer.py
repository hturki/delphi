import grpc
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import Int32Value, BoolValue

from delphi.coordinator import Coordinator
from delphi.proto.admin_pb2_grpc import AdminServiceServicer
from delphi.utils import log_exceptions_and_abort


class AdminServicer(AdminServiceServicer):

    def __init__(self, coordinator: Coordinator):
        self._coordinator = coordinator

    @log_exceptions_and_abort
    def Reset(self, request: BoolValue, context: grpc.ServicerContext) -> Empty:
        self._coordinator.reset(request.value)
        return Empty()

    @log_exceptions_and_abort
    def GetLastTrainedVersion(self, request: Empty, context: grpc.ServicerContext) -> Int32Value:
        return Int32Value(value=self._coordinator.get_last_trained_version())
