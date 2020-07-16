import threading

from delphi.proto.learning_module_pb2 import SearchId
from delphi.search import Search


class SearchManager(object):

    def __init__(self):
        self._lock = threading.Lock()
        self._searches = {}

    def set_search(self, search_id: SearchId, search: Search) -> None:
        with self._lock:
            assert search_id not in self._searches
            self._searches[search_id] = search

    def get_search(self, search_id: SearchId) -> Search:
        with self._lock:
            return self._searches[search_id]

    def remove_search(self, search_id: SearchId) -> Search:
        with self._lock:
            search = self._searches[search_id]
            del self._searches[search_id]

        return search
