import os

import torch
import torch.distributed as dist
from google.protobuf.any_pb2 import Any

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model_trainer import TrainingStyle, DataRequirement
from delphi.model_trainer_base import ModelTrainerBase


class PytorchTrainerBase(ModelTrainerBase):

    def __init__(self, context: ModelTrainerContext, distributed: bool):
        super().__init__()
        self.context = context
        self.distributed = distributed

        if self.distributed:
            os.environ['NCCL_DEBUG'] = 'INFO'
            os.environ['NCCL_SOCKET_IFNAME'] = 'eno2.1002'
            os.environ['MASTER_ADDR'] = self.context.nodes[0].url.split(':')[0]
            os.environ['MASTER_PORT'] = str(self.context.port - 1)

            # initialize the process group
            dist.init_process_group('nccl', rank=self.context.node_index, world_size=len(self.context.nodes))

            # Explicitly setting seed to make sure that models created in two processes
            # start from same random weights and biases.
            torch.manual_seed(42)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def data_requirement(self) -> DataRequirement:
        return DataRequirement.DISTRIBUTED_FULL if self.distributed else DataRequirement.MASTER_ONLY

    @property
    def training_style(self) -> TrainingStyle:
        return TrainingStyle.DISTRIBUTED if self.distributed else TrainingStyle.MASTER_ONLY

    @property
    def should_sync_model(self) -> bool:
        return not self.distributed

    def message_internal(self, request: Any) -> Any:
        pass
