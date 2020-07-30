from typing import Mapping

from delphi.attribute_provider import AttributeProvider


class SimpleAttributeProvider(AttributeProvider):

    def __init__(self, attributes: Mapping[str, bytes]):
        self._attributes = attributes

    def get(self) -> Mapping[str, bytes]:
        return self._attributes
