from typing import Optional

from delphi.object_provider import ObjectProvider


class ProviderAndResult(object):

    def __init__(self, provider: ObjectProvider, label: str, score: Optional[float], model_version: Optional[int]):
        self.provider = provider
        self.label = label
        self.score = score
        self.model_version = model_version
