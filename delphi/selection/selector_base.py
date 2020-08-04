import queue
import threading
from abc import abstractmethod
from typing import Optional

from delphi.result_provider import ResultProvider
from delphi.selection.selector import Selector


class SelectorBase(Selector):

    def __init__(self):
        self.result_queue = queue.Queue(maxsize=500)
        self.stats_lock = threading.Lock()
        self.items_processed = 0

        self._finish_event = threading.Event()
        self._model_present = False

    @abstractmethod
    def add_result_inner(self, result: ResultProvider) -> None:
        pass

    def add_result(self, result: ResultProvider) -> None:
        if not self._model_present:
            self.result_queue.put(result)
        else:
            self.add_result_inner(result)

        with self.stats_lock:
            self.items_processed += 1

    def finish(self) -> None:
        self._finish_event.set()
        self.result_queue.put(None)

    def get_result(self) -> Optional[ResultProvider]:
        while True:
            try:
                return self.result_queue.get(timeout=10)
            except queue.Empty:
                if self._finish_event.is_set():
                    return None
