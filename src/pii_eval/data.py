from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from datasets import Dataset, concatenate_datasets
import datasets

from pii_eval.util import (
    LabelSpan,
    TextSpan,
    autoinit,
    to_dict_recursive,
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

    def to_concrete(self) -> "PiiEntryConcrete":
        return PiiEntryConcrete(
            text_=self.text, spans_=self.spans, bio_=self.bio, extra_=self.extra
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "text": self.text,
            "spans": [asdict(x) for x in self.spans],
            "bio": [asdict(x) for x in self.bio],
            "extra": to_dict_recursive(self.extra),
        }


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
    extra_f: Callable[[], dict[str, Any]] = lambda: {}

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
        return self.extra_f()


@autoinit
class PiiDataset(ABC):
    name: str
    metadata: dict[str, Any] = {}

    @abstractmethod
    def __getitem__(self, key) -> PiiEntry: ...

    @abstractmethod
    def __iter__(self) -> Iterator[PiiEntry]: ...  # TODO typehint -> ?

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def dump(self, path: Path): ...

    @cached_property
    def uid_to_idx(self):
        return {x.uid: i for i, x in enumerate(self)}

    def get_uid(self, uid: int | str) -> PiiEntry:
        return self[self.uid_to_idx(uid)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entries": [x.to_dict() for x in self],
            "metadata": to_dict_recursive(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def save_json(self, path: Path) -> str:
        # path.parent.mkdir(parents=True, exist_ok=True) # TODO: not necessary with json.dump?
        with path.open("w") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


@dataclass
class FromEntries(PiiDataset):
    entries: list[PiiEntry]

    def __getitem__(self, key) -> PiiEntry:
        return self.entries[key]

    def __iter__(self) -> Iterator[PiiEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def load_json(path: Path) -> "FromEntries":
        raise NotImplementedError  # TODO


@dataclass
class FromDatasets(PiiDataset):
    load_dataset: Callable[[], Dataset]
    from_row: Callable[[Any], PiiEntry]

    @cached_property
    def dataset(self) -> Dataset:
        return self.load_dataset()

    def __getitem__(self, key) -> PiiEntry:
        return self.from_row(self.dataset[key])

    def __len__(self) -> int:
        return len(self.dataset)

    def dump(self, path: Path):
        raise NotImplementedError

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


class PiiModel(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(
        self, entry: PiiEntry, label_map: dict[str, str] | None = None
    ) -> PiiEntry:
        pass
