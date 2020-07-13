import queue
from typing import List

from delphi.model import Model
from delphi.selection.reexamination_strategy import ReexaminationStrategy


class NoReexaminationStrategy(ReexaminationStrategy):

    def reexamine(self, model: Model, queues: List[queue.PriorityQueue]):
        pass