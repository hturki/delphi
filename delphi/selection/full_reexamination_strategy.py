import queue
from typing import List

from delphi.model import Model
from delphi.selection.reexamination_strategy import ReexaminationStrategy


class FullReexaminationStrategy(ReexaminationStrategy):

    def reexamine(self, model: Model, queues: List[queue.PriorityQueue]):
        to_reexamine = []
        for priority_queue in queues[:-1]:
            while True:
                try:
                    to_reexamine.append(priority_queue.get_nowait()[1])
                except queue.Empty:
                    break

        for result in model.infer(to_reexamine):
            queues[-1].put((-result.score, result.id, result))
