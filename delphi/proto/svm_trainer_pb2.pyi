# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Optional as typing___Optional,
    Text as typing___Text,
    Union as typing___Union,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int
if sys.version_info < (3,):
    builtin___buffer = buffer
    builtin___unicode = unicode


class SVMTrainerMessage(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def setTrainResult(self) -> global___SetTrainResult: ...

    @property
    def setParamGrid(self) -> global___SetParamGrid: ...

    def __init__(self,
        *,
        setTrainResult : typing___Optional[global___SetTrainResult] = None,
        setParamGrid : typing___Optional[global___SetParamGrid] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> SVMTrainerMessage: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> SVMTrainerMessage: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"setParamGrid",b"setParamGrid",u"setTrainResult",b"setTrainResult",u"value",b"value"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"setParamGrid",b"setParamGrid",u"setTrainResult",b"setTrainResult",u"value",b"value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions___Literal[u"value",b"value"]) -> typing_extensions___Literal["setTrainResult","setParamGrid"]: ...
global___SVMTrainerMessage = SVMTrainerMessage

class SetTrainResult(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    version = ... # type: builtin___int
    params = ... # type: typing___Text
    score = ... # type: builtin___float
    model = ... # type: builtin___bytes

    def __init__(self,
        *,
        version : typing___Optional[builtin___int] = None,
        params : typing___Optional[typing___Text] = None,
        score : typing___Optional[builtin___float] = None,
        model : typing___Optional[builtin___bytes] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> SetTrainResult: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> SetTrainResult: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"model",b"model",u"params",b"params",u"score",b"score",u"version",b"version"]) -> None: ...
global___SetTrainResult = SetTrainResult

class SetParamGrid(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    grid = ... # type: typing___Text

    def __init__(self,
        *,
        grid : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> SetParamGrid: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> SetParamGrid: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"grid",b"grid"]) -> None: ...
global___SetParamGrid = SetParamGrid