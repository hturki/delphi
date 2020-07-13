from typing import Iterator

from opendiamond.client.search import DiamondSearch
from opendiamond.server.object_ import ATTR_OBJ_ID, ATTR_DATA

from delphi.proto.learning_module_pb2 import InferObject
from delphi.retrieval.retriever import Retriever


class DiamondRetriever(Retriever):

    def __init__(self, search: DiamondSearch):
        self._search = search

    def start(self) -> None:
        self._search.start()

    def stop(self) -> None:
        self._search.close()

    def get_objects(self) -> Iterator[InferObject]:
        for result in self._search.results:
            yield InferObject(objectId=result[ATTR_OBJ_ID], content=result[ATTR_DATA])
