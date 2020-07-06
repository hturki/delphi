import grpc

from delphi.proto.internal_pb2_grpc import InternalServiceStub
from delphi.proto.learning_module_pb2_grpc import LearningModuleServiceStub


class LearningModuleStub(object):

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        channel = grpc.insecure_channel('{}:{}'.format(host, port), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
        ])

        self.api = LearningModuleServiceStub(channel)
        self.internal = InternalServiceStub(channel)
