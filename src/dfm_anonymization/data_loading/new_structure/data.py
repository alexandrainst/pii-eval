from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from datasets import Dataset

from dfm_anonymization.data_loading.new_structure.util import (
    LabelSpan,
    TextSpan,
    spans_to_bio,
)


class PiiEntry(ABC):
    @property
    @abstractmethod
    def text(self) -> str: ...

    @property
    @abstractmethod
    def spans(self) -> list[TextSpan]: ...

    @property
    @abstractmethod
    def bio(self) -> list[LabelSpan]: ...

    @property
    def extra(self) -> dict[str, Any]:
        return {}


@dataclass
class PiiEntryConcrete(PiiEntry):
    text: str
    spans: list[TextSpan]
    bio: list[LabelSpan]
    extra: dict[str, Any]


@dataclass
class PiiEntryLazy(PiiEntry):
    text_f: Callable[[], str]
    spans_f: Callable[[], list[TextSpan]]
    bio_f: Callable[[], list[LabelSpan]]
    extra_f: Callable[[], dict[str, Any]] | None = None

    @property
    def text(self) -> str:
        return self.text_f()

    @property
    def spans(self) -> list[LabelSpan]:
        return self.spans_f()

    @property
    def bio(self) -> list[LabelSpan]:
        return self.bio_f()

    @property
    def extra(self) -> dict[str, Any]:
        if self.extra_f:
            return self.extra_f()
        else:
            return super().extra

    def to_concrete(self) -> PiiEntryConcrete:
        PiiEntryConcrete(
            text=self.text, spans=self.spans, bio=self.bio, extra=self.extra
        )


@dataclass
class PiiDataset(ABC):
    name: str

    @abstractmethod
    def __getitem__(self, key) -> PiiEntry:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@dataclass
class SpansDatasets(PiiDataset):
    load_dataset: Callable[[], Dataset]
    from_row: Callable[[Any], PiiEntry]

    _loaded_dataset: Dataset | None = None

    @property
    def dataset(self) -> Dataset:
        if not self._loaded_dataset:
            self._loaded_dataset = self.load_dataset()
        return self._loaded_dataset

    def __getitem__(self, key) -> PiiEntry:
        return self.from_row(self.dataset[key])

    def __len__(self) -> int:
        return len(self.dataset)
