import queue
from typing import List

from delphi.model import Model
from delphi.selection.reexamination_strategy import ReexaminationStrategy


class TopReexaminationStrategy(ReexaminationStrategy):

    def __init__(self, k: int):
        self._k = k

    def reexamine(self, model: Model, queues: List[queue.PriorityQueue]):
        to_reexamine = []
        for priority_queue in queues[:-1]:
            for _ in range(self._k):
                try:
                    to_reexamine.append(priority_queue.get_nowait()[1])
                except queue.Empty:
                    break

        for result in model.infer(to_reexamine):
            queues[-1].put((-result.score, result.id, result))
