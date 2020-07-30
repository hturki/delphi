from abc import ABCMeta, abstractmethod
from typing import Iterable, Optional

from delphi.model import Model
from delphi.result_provider import ResultProvider


class Selector(metaclass=ABCMeta):

    @abstractmethod
    def add_result(self, result: ResultProvider) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass

    @abstractmethod
    def get_results(self) -> Iterable[ResultProvider]:
        pass

    @abstractmethod
    def new_model(self, model: Optional[Model]) -> None:
        pass
