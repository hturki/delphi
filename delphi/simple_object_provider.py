from typing import Mapping

from delphi.object_provider import ObjectProvider


class SimpleObjectProvider(ObjectProvider):

    def __init__(self, id: str, content: bytes, attributes: Mapping[str, bytes]):
        self._id = id
        self._content = content
        self._attributes = attributes

    @property
    def id(self) -> str:
        return self._id

    def get_content(self) -> bytes:
        return self._content

    def get_attributes(self) -> Mapping[str, bytes]:
        return self._attributes
