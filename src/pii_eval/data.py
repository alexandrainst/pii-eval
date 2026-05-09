from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Literal

from datasets import Dataset, concatenate_datasets
import datasets

from pii_eval.util import (
    LabelSpan,
    TextSpan,
)


class PiiEntry(ABC):
    @property
    @abstractmethod
    def uid(self) -> int | str: ...

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
    uid_: int | str
    text_: str
    spans_: list[TextSpan]
    bio_: list[LabelSpan]
    extra_: dict[str, Any]

    @property
    def uid(self) -> int | str:
        return self.uid_

    @property
    def text(self) -> str:
        return self.text_

    @property
    def spans(self) -> list[TextSpan]:
        return self.spans_

    @property
    def bio(self) -> list[LabelSpan]:
        return self.bio_

    @property
    def extra(self) -> dict[str, Any]:
        return self.extra_


@dataclass
class PiiEntryLazy(PiiEntry):
    uid_f: Callable[[], int | str]
    text_f: Callable[[], str]
    spans_f: Callable[[], list[TextSpan]]
    bio_f: Callable[[], list[LabelSpan]]
    extra_f: Callable[[], dict[str, Any]] | None = None

    @property
    def uid(self) -> str:
        return self.uid_f()

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
        return PiiEntryConcrete(
            text_=self.text, spans_=self.spans, bio_=self.bio, extra_=self.extra
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
class FromDatasets(PiiDataset):
    load_dataset: Callable[[], Dataset]
    from_row: Callable[[Any], PiiEntry]

    _uid_to_idx: dict[int | str, int] | None = None

    @cached_property
    def dataset(self) -> Dataset:
        return self.load_dataset()

    @cached_property
    def uid_to_idx(self):
        return {x.uid: i for i, x in enumerate(self)}

    def __getitem__(self, key) -> PiiEntry:
        return self.from_row(self.dataset[key])

    def __len__(self) -> int:
        return len(self.dataset)

    def get_uid(self, uid: int | str) -> PiiEntry:
        return self[self.uid_to_idx(uid)]

    @staticmethod
    def from_hf_tag(
        hf_tag: str,
        from_row: Callable[[Any], PiiEntry],
        splits: list[str] | Literal["all"] = "all",
    ):
        def load_dataset() -> Dataset:
            ds_dict = datasets.load_dataset(hf_tag)
            if splits == "all":
                return concatenate_datasets(ds_dict.values())
            else:
                return concatenate_datasets([ds_dict[x] for x in splits])

        return FromDatasets(
            name=hf_tag.split("/")[-1],
            load_dataset=load_dataset,
            from_row=from_row,
        )
