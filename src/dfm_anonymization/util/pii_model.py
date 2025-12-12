
from abc import ABC, abstractmethod
from typing import Any, Dict

type PiiModelResult = Dict[str, Any]


class PiiModel(ABC):
    @abstractmethod
    def __call__(self, *args, **kwds) -> PiiModelResult:
        pass
