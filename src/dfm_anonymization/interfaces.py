from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set, Tuple

from dfm_anonymization.dataset import PiiDataEntry


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
        # print(f"PiiModel '{name}' initialized")

    @abstractmethod
    def __call__(
        self, entry: PiiDataEntry, label_map: dict[str, str] | None = None
    ) -> List[str]:
        pass
