import io
from typing import Mapping, Dict, Union

from PIL import Image
from opendiamond.attributes import IntegerAttributeCodec

from delphi.attribute_provider import AttributeProvider

INT_CODEC = IntegerAttributeCodec()


class DiamondAttributeProvider(AttributeProvider):

    def __init__(self, attributes: Dict[str, bytes], image_provider: Union[str, bytes]):
        self._attributes = attributes

        for attribute in ['_rows.int', '_cols.int', 'thumbnail.jpeg']:
            if attribute in self._attributes:
                del self._attributes[attribute]

        self._image_provider = image_provider

    def get(self) -> Mapping[str, bytes]:
        attributes = dict(self._attributes)

        image = Image.open(self._image_provider)
        width, height = image.size

        attributes['_rows.int'] = INT_CODEC.encode(height)
        attributes['_cols.int'] = INT_CODEC.encode(width)

        thumbnail = io.BytesIO()
        image.resize((200, 150)).save(thumbnail, 'JPEG')
        attributes['thumbnail.jpeg'] = thumbnail.getvalue()

        return attributes
