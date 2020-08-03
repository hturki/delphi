import io
import threading
from pathlib import Path
from typing import Iterable, Sized

from logzero import logger
from opendiamond.attributes import StringAttributeCodec
from opendiamond.client.search import DiamondSearch, FilterSpec, Blob
from opendiamond.config import DiamondConfig
from opendiamond.protocol import XDR_reexecute
from opendiamond.scope import ScopeCookie
from opendiamond.server.object_ import ATTR_OBJ_ID, ATTR_DATA, ATTR_DEVICE_NAME

from delphi.object_provider import ObjectProvider
from delphi.proto.learning_module_pb2 import DelphiObject, DiamondDataset
from delphi.retrieval.diamond_attribute_provider import DiamondAttributeProvider
from delphi.retrieval.retriever import Retriever
from delphi.retrieval.retriever_stats import RetrieverStats

STRING_CODEC = StringAttributeCodec()


class DiamondRetriever(Retriever):

    def __init__(self, dataset: DiamondDataset):
        self._dataset = dataset
        self._search = self._create_search()
        self._start_event = threading.Event()

        try:
            self._diamond_config = DiamondConfig()
        except Exception as e:
            logger.info('No local diamond config found')
            logger.exception(e)
            self._diamond_config = None

    def start(self) -> None:
        self._search.start()
        self._start_event.set()

    def stop(self) -> None:
        self._search.close()

    def get_objects(self) -> Iterable[ObjectProvider]:
        for result in self._search.results:
            content = result[ATTR_DATA]
            del result[ATTR_DATA]
            object_id = STRING_CODEC.decode(result[ATTR_OBJ_ID])

            # Optimization to directly load the data from local disk if possible instead of holding it in memory
            if self._diamond_config is not None \
                    and STRING_CODEC.decode(result[ATTR_DEVICE_NAME]) in self._diamond_config.serverids \
                    and object_id.startswith('http://localhost'):
                image_provider = Path(self._diamond_config.dataroot) / object_id.split('/collection/id/')[1]
            else:
                image_provider = io.BytesIO(content)

            yield ObjectProvider(object_id, content, DiamondAttributeProvider(result, image_provider))

    def get_object(self, object_id: str, attributes: Sized) -> DelphiObject:
        # Each Delphi server should be connected to only one Diamond server
        conn = next(iter(self._search._connections.values()))

        # Send reexecute request
        request = XDR_reexecute(object_id=object_id, attrs=attributes if len(attributes) > 1 else None)
        reply = conn.control.reexecute_filters(request)

        # Return object attributes
        dct = dict((attr.name, attr.value) for attr in reply.attrs)
        content = dct[ATTR_DATA]
        del dct[ATTR_DATA]

        return DelphiObject(objectId=object_id, content=content, attributes=dct)

    def get_stats(self) -> RetrieverStats:
        self._start_event.wait()

        stats = self._search.get_stats()
        return RetrieverStats(stats['objs_total'], stats['objs_dropped'], stats['objs_false_negative'])

    def _create_search(self) -> DiamondSearch:
        return DiamondSearch([ScopeCookie.parse(x) for x in self._dataset.cookies],
                             [FilterSpec(x.name, Blob(x.code), x.arguments, Blob(x.blob), x.dependencies,
                                         x.minScore, x.maxScore) for x in self._dataset.filters],
                             False,
                             list(self._dataset.attributes) + [ATTR_DATA])
