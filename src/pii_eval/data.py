# from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import datasets
from datasets import Dataset, concatenate_datasets

from pii_eval.span import LabelSpan, bio_to_spans, spans_to_bio
from pii_eval.util import to_dict_recursive


@dataclass
class PiiEntry(ABC):
    @property
    @abstractmethod
    def uid(self) -> int | str: ...

    @property
    @abstractmethod
    def text(self) -> str: ...

    @property
    @abstractmethod
    def spans(self) -> list[LabelSpan]: ...

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
    spans_: list[LabelSpan]
    bio_: list[LabelSpan]
    extra_: dict[str, Any]

    @property
    def uid(self) -> int | str:
        return self.uid_

    @property
    def text(self) -> str:
        return self.text_

    @property
    def spans(self) -> list[LabelSpan]:
        return self.spans_

    @property
    def bio(self) -> list[LabelSpan]:
        return self.bio_

    @property
    def extra(self) -> dict[str, Any]:
        return self.extra_

    @staticmethod
    def from_dict(data: dict) -> PiiEntry:
        text = data["text"]
        spans = data.get("spans")
        bio = data.get("bio")
        if bio:
            bio = [LabelSpan(**x) for x in bio]
        if spans:
            spans = [LabelSpan(**x) for x in spans]
        tokens = data.get("tokens")
        bio_tags = data.get("bio_tags")
        if not spans and not bio and tokens and bio_tags:
            bio = LabelSpan.from_text_and_token(
                text=text, tokens=tokens, labels=bio_tags
            )
        if not spans:
            spans = bio_to_spans(text=text, bio=bio)
        if not bio:
            bio = spans_to_bio(text=text, spans=spans)
        return PiiEntryConcrete(
            uid_=data.get("uid") or data.get("id"),
            text_=text,
            spans_=spans,
            bio_=bio,
            extra_=data.get("extra", {}),
        )


@dataclass
class PiiEntryLazy(PiiEntry):
    uid_f: Callable[[], int | str]
    text_f: Callable[[], str]
    spans_f: Callable[[], list[LabelSpan]]
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


@dataclass
class PiiDataset(ABC):
    name: str
    metadata: dict[str, Any]  #  = field(default_factory=dict)

    @abstractmethod
    def __getitem__(self, key) -> PiiEntry: ...

    @abstractmethod
    def __iter__(self) -> Iterator[PiiEntry]: ...  # TODO typehint -> ?

    @abstractmethod
    def __len__(self) -> int: ...

    @cached_property
    def uid_to_idx(self):
        return {x.uid: i for i, x in enumerate(self)}

    def get(self, key, default=None) -> PiiEntry | None:
        try:
            return self[key]
        except:
            return default

    def get_uid(self, uid: int | str, default=None) -> PiiEntry:
        return self.get(self.uid_to_idx.get(uid), default=default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entries": [x.to_dict() for x in self],
            "metadata": to_dict_recursive(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save_json(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


@dataclass
class FromEntries(PiiDataset):
    entries: list[PiiEntry]  #  = field(default=list)

    def __getitem__(self, key) -> PiiEntry:
        return self.entries[key]

    def __iter__(self) -> Iterator[PiiEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def from_dict(data: dict, name: str) -> "FromEntries":
        return FromEntries(
            name=name,
            metadata=data.get("metadata", {}),
            entries=[
                x if type(x) == PiiEntry else PiiEntryConcrete.from_dict(x)
                for x in data["entries"]
            ],
        )

    @staticmethod
    def load_json(path: Path, name: str | None = None) -> "FromEntries":
        with open(path, mode="r") as f:
            data: dict = json.load(f)
        return FromEntries.from_dict(data, name=name if name else path.stem)


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
