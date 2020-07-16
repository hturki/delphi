from typing import Iterator, Iterable

from opendiamond.client.search import DiamondSearch
from opendiamond.protocol import XDR_reexecute
from opendiamond.server.object_ import ATTR_OBJ_ID, ATTR_DATA

from delphi.proto.learning_module_pb2 import DelphiObject
from delphi.retrieval.retriever import Retriever


class DiamondRetriever(Retriever):

    def __init__(self, search: DiamondSearch):
        self._search = search

    def start(self) -> None:
        self._search.start()

    def stop(self) -> None:
        self._search.close()

    def get_objects(self) -> Iterator[DelphiObject]:
        for result in self._search.results:
            yield DelphiObject(objectId=result[ATTR_OBJ_ID], content=result[ATTR_DATA])

    def get_object(self, object_id: str, attributes: Iterable[str]) -> DelphiObject:
        conn = self._search._connections[0]  # Each Delphi server should be connected to only one Diamond server
        conn.connect()
        conn.setup(next(iter(self._search._cookie_map.values())), self._search._filters)

        # Send reexecute request
        request = XDR_reexecute(object_id=object_id, attrs=attributes)
        reply = conn.control.reexecute_filters(request)

        # Return object attributes
        dct = dict((attr.name, attr.value) for attr in reply.attrs)
        content = dct[ATTR_DATA]
        del dct[ATTR_DATA]
        return DelphiObject(objectId=object_id, content=content, attributes=conn._sanitize_obj_dict(dct))
