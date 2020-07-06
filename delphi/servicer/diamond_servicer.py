import threading

import grpc
from google.protobuf.wrappers_pb2 import BoolValue

from delphi.coordinator import Coordinator
from delphi.proto.diamond_pb2 import FilterRole
from delphi.proto.diamond_pb2_grpc import DiamondServiceServicer
from delphi.utils import log_exceptions_and_abort


class DiamondServicer(DiamondServiceServicer):

    def __init__(self, coordinator: Coordinator):
        self._coordinator = coordinator
        self._init_lock = threading.Lock()
        self._got_first_filter_role_request = False

    # This is a hack since we have multiple filter threads all hitting the same learning module. This is used so that:
    # - Only one thread across all search nodes sends over the example zip (since it's identical across all filter
    # threads)
    # - Only one filter thread across all search nodes asks for model exports
    @log_exceptions_and_abort
    def FilterInit(self, request: BoolValue, context: grpc.ServicerContext) -> FilterRole:
        send_examples = False
        request_export = False

        with self._init_lock:
            if not self._got_first_filter_role_request:
                self._got_first_filter_role_request = True
                if request.value:
                    self._coordinator.has_initial_examples = True
                    send_examples = self._coordinator.node_index == 0
                request_export = self._coordinator.node_index == 0

        return FilterRole(sendExamples=send_examples, requestExport=request_export)
