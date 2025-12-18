

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from dfm_anonymization.inference.dataset import PiiDataEntry


class Tokenizer(ABC):
    @abstractmethod
    def __call__(self, words: List[str]) -> Tuple[Any, List[Optional[int]]]:
        pass


class TokenClassifier(ABC):
    @abstractmethod
    def __call__(self, tokens: Any) -> List[str]:
        pass


class PiiModel(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, entry: PiiDataEntry) -> List[str]:
        pass
