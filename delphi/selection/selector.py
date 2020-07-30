from abc import ABCMeta, abstractmethod
from typing import Iterable, Optional

from delphi.model import Model
from delphi.provider_and_result import ProviderAndResult


class Selector(metaclass=ABCMeta):

    @abstractmethod
    def add_result(self, result: ProviderAndResult) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass

    @abstractmethod
    def get_results(self) -> Iterable[ProviderAndResult]:
        pass

    @abstractmethod
    def new_model(self, model: Optional[Model]) -> None:
        pass
