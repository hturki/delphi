import grpc
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import Int32Value

from delphi.proto.admin_pb2 import ResetRequest
from delphi.proto.admin_pb2_grpc import AdminServiceServicer
from delphi.proto.search_pb2 import SearchId
from delphi.search_manager import SearchManager
from delphi.utils import log_exceptions_and_abort


class AdminServicer(AdminServiceServicer):

    def __init__(self, manager: SearchManager):
        self._manager = manager

    @log_exceptions_and_abort
    def Reset(self, request: ResetRequest, context: grpc.ServicerContext) -> Empty:
        self._manager.get_search(request.searchId).reset(request.trainOnly)
        return Empty()

    @log_exceptions_and_abort
    def GetLastTrainedVersion(self, request: SearchId, context: grpc.ServicerContext) -> Int32Value:
        return Int32Value(value=self._manager.get_search(request).get_last_trained_version())
