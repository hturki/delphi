import math
import queue
from typing import Iterator

from delphi.model import Model
from delphi.proto.learning_module_pb2 import InferResult
from delphi.selection.reexamination_strategy import ReexaminationStrategy
from delphi.selection.selector import Selector
from delphi.utils import to_iter


class TopKSelector(Selector):

    def __init__(self, k: int, batch_size: int, reexamination_strategy: ReexaminationStrategy):
        assert k < batch_size
        self._k = k
        self._batch_size = batch_size
        self._reexamination_strategy = reexamination_strategy

        self._priority_queues = [queue.PriorityQueue()]
        self._batch_added = 0

        self._result_queue = queue.Queue()
        self._result_iterator = to_iter(self._result_queue)

    def add_result(self, result: InferResult) -> None:
        self._priority_queues[-1].put((-result.score, result))
        self._batch_added += 1
        if self._batch_added == self._batch_size:
            for _ in range(self._k):
                self._result_queue.put(self._priority_queues[-1].get()[1])
            self._batch_added = 0

    def get_results(self) -> Iterator[InferResult]:
        yield from self._result_iterator

    def new_model(self, model: Model) -> None:
        # add fractional batch before possibly discarding results in old queue
        for _ in range(math.ceil(float(self._k) * self._batch_added / self._batch_size)):
            self._result_queue.put(self._priority_queues[-1].get()[1])

        self._priority_queues.append(queue.PriorityQueue())
        self._reexamination_strategy.reexamine(model, self._priority_queues)
        self._batch_added = 0