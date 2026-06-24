# from __future__ import annotations
import json
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import datasets
from datasets import Dataset, concatenate_datasets

from pii_eval.span import LabelSpan
from pii_eval.util import to_dict_recursive


@dataclass
class PiiEntry:
    uid: int | str
    text: str
    bio: list[LabelSpan]
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "PiiEntry":
        valid_keys = {f.name for f in fields(cls)}
        filtered_d = {k: v for k, v in d.items() if k in valid_keys}
        filtered_d["bio"] = [LabelSpan(**x) for x in filtered_d["bio"]]
        return PiiEntry(**filtered_d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "text": self.text,
            "bio": [asdict(x) for x in self.bio],
            "meta": to_dict_recursive(self.meta),
        }


@dataclass
class PiiDataset:
    name: str
    meta: dict[str, Any] = field(default_factory=dict)
    entries: list[PiiEntry] = field(default_factory=list)

    def __getitem__(self, key) -> PiiEntry:
        return self.entries[key]

    @abstractmethod
    def __iter__(self) -> Iterator[PiiEntry]:
        return iter(self.entries)

    @abstractmethod
    def __len__(self) -> int:
        return len(self.entries)

    def get_samples(self, samples: int):
        return random.sample(self.entries, min(samples, len(self)))

    @cached_property
    def uid_to_idx(self):
        return {x.uid: i for i, x in enumerate(self)}

    def get(self, index: int, default=None) -> PiiEntry:
        return self.entries[index] if index < len(self) else default

    def get_uid(self, uid: int | str, default=None) -> PiiEntry:
        return self.get(self.uid_to_idx.get(uid), default=default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "meta": to_dict_recursive(self.meta),
            "entries": [x.to_dict() for x in self],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save_json(self, path: Path | str) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @staticmethod
    def from_dict(data: dict, name: str) -> "PiiDataset":
        return PiiDataset(
            name=name,
            meta=data.get("meta", {}),
            entries=[
                x if type(x) == PiiEntry else PiiEntry.from_dict(x)
                for x in data["entries"]
            ],
        )

    @staticmethod
    def load_json(path: Path, name: str | None = None) -> "PiiDataset":
        with open(path, mode="r") as f:
            data: dict = json.load(f)
        return PiiDataset.from_dict(data, name=name if name else path.stem)

    @staticmethod
    def load_old_format(path: Path, name: str | None = None) -> "PiiDataset":
        with open(path, mode="r") as f:
            data: dict = json.load(f)
        assert "entries" in data
        entries = data["entries"]
        assert set(["text", "tokens", "bio_tags", "id"]).issubset(entries[0].keys())
        new_entries = []
        for entry in entries:
            text = entry["text"]
            tokens = entry["tokens"]
            bio_tags = entry["bio_tags"]
            id = entry["id"]
            new_entries.append(
                PiiEntry(
                    uid=id,
                    text=text,
                    bio=LabelSpan.from_text_and_tokens(
                        text=text,
                        tokens=tokens,
                        labels=bio_tags,
                        include_o_labels=False,
                    ),
                    meta={},
                )
            )
        return PiiDataset(
            name=name if name else path.stem, entries=new_entries, meta={}
        )

    @staticmethod
    def load_tokens(path: Path, name: str | None = None) -> "PiiDataset":
        with open(path, mode="r") as f:
            data: dict = json.load(f)
        return PiiDataset.from_dict(data, name=name if name else path.stem)

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

        return PiiDataset(
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
