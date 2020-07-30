import math
import queue
import threading
from typing import Optional, Iterable

from delphi.model import Model
from delphi.result_provider import ResultProvider
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
        self._insert_lock = threading.Lock()

        self._result_queue = queue.Queue(maxsize=500)
        self._result_iterator = to_iter(self._result_queue)
        self._model_present = False

    def add_result(self, result: ResultProvider) -> None:
        with self._insert_lock:
            if not self._model_present:
                self._result_queue.put(result)
            else:
                self._priority_queues[-1].put((-result.score, result.id, result))
                self._batch_added += 1
                if self._batch_added == self._batch_size:
                    for _ in range(self._k):
                        self._result_queue.put(self._priority_queues[-1].get()[-1])
                    self._batch_added = 0

    def finish(self) -> None:
        self._result_queue.put(None)

    def get_results(self) -> Iterable[ResultProvider]:
        yield from self._result_iterator

    def new_model(self, model: Optional[Model]) -> None:
        with self._insert_lock:
            self._model_present = model is not None

            if self._model_present:
                # add fractional batch before possibly discarding results in old queue
                for _ in range(math.ceil(float(self._k) * self._batch_added / self._batch_size)):
                    self._result_queue.put(self._priority_queues[-1].get()[1])

                self._priority_queues.append(queue.PriorityQueue())
                self._reexamination_strategy.reexamine(model, self._priority_queues)
            else:
                # this is a reset, discard everything
                self._priority_queues = [queue.PriorityQueue()]

            self._batch_added = 0
