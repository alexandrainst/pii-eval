import hashlib
import os
from pathlib import Path
from typing import Any, Callable

import datasets
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from rich.progress import track

from pii_eval.constants import PII_EVAL_CACHE_PATH
from pii_eval.data import (
    PiiDataset,
    PiiEntry,
)
from pii_eval.span import LabelSpan, spans_to_bio
from pii_eval.util import count_labels_from_entries


def _cache_path_for(name: str) -> Path:
    return Path(PII_EVAL_CACHE_PATH) / f"datasets/{name}.pickle"


def _load_hf_token() -> str | None:
    # Resolve .env relative to repository root (src/pii_eval/load.py -> repo root).
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    return os.getenv("HF_TOKEN")


def _labelspan_from_dict(span: dict[str, Any], text: str) -> LabelSpan:
    start = int(span.get("start", 0))
    end = int(span.get("end", 0))
    return LabelSpan(
        start=start,
        end=end,
        text=str(span.get("text", text[start:end])),
        label=str(span.get("label", "O")),
    )


def _safe_int_uid(row: dict[str, Any], text: str) -> int:
    explicit = row.get("uid", row.get("id"))
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def _dataset_from_hf(
    *,
    hf_tag: str,
    name: str,
    from_row,
    token: str | None = None,
    splits: list[str] | None = None,
) -> PiiDataset:
    ds_dict = datasets.load_dataset(hf_tag, token=token)
    ds: Dataset = concatenate_datasets(
        ds_dict.values() if splits is None else [ds_dict[x] for x in splits]
    )
    entries: list[PiiEntry] = []
    for row in track(ds, transient=True, description=f"Loading {name} ..."):
        entry = from_row(row)
        if entry is not None:
            entries.append(entry)
    return PiiDataset(
        name=name,
        entries=entries,
        meta={"hf_tag": hf_tag, "labels": count_labels_from_entries(entries)},
    )


def load_dane() -> PiiDataset:
    hf_tag = "jharder/dane-copy"
    name = "dane"
    cache_path = _cache_path_for(name)

    if cache_path.exists():
        return PiiDataset.load_pickle(cache_path)

    def from_row(row: dict[str, Any]) -> PiiEntry:
        text = str(row.get("text", ""))
        raw_bio = row.get("bio", [])
        bio = [_labelspan_from_dict(span, text) for span in raw_bio]
        return PiiEntry(uid=_safe_int_uid(row, text=text), text=text, bio=bio, meta={})

    ds = _dataset_from_hf(hf_tag=hf_tag, name=name, from_row=from_row)

    ds.dump_pickle(cache_path)
    return ds


def load_dansk_ner() -> PiiDataset:
    hf_tag = "chcaa/dansk-ner"
    name = "dansk-ner"
    cache_path = _cache_path_for(name)

    if cache_path.exists():
        return PiiDataset.load_pickle(cache_path)

    def from_row(row: dict[str, Any]) -> PiiEntry | None:
        text = str(row.get("text", ""))
        spans = [
            LabelSpan(
                start=int(x["start"]),
                end=int(x["end"]),
                text=text[int(x["start"]) : int(x["end"])],
                label=str(x["label"]),
            )
            for x in row.get("ents", [])
            if isinstance(x, dict)
        ]
        bio = spans_to_bio(text=text, spans=spans)
        extra_features = [
            "dagw_source",
            "dagw_domain",
            "dagw_source_full",
        ]
        return PiiEntry(
            uid=_safe_int_uid(row, text=text),
            text=text,
            bio=bio,
            meta={
                feature: row[feature] for feature in extra_features if feature in row
            },
        )

    ds = _dataset_from_hf(hf_tag=hf_tag, name=name, from_row=from_row)
    ds.dump_pickle(cache_path)
    return ds


def load_dimmi() -> PiiDataset:
    hf_tag = "jharder/dimmi"
    name = "dimmi"
    cache_path = _cache_path_for(name)

    if cache_path.exists():
        return PiiDataset.load_pickle(cache_path)

    hf_token = _load_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required to load private dataset 'jharder/dimmi'."
        )

    def from_row(row: dict[str, Any]) -> PiiEntry:
        text = str(row.get("text", ""))
        bio = [_labelspan_from_dict(span, text) for span in row.get("bio", [])]
        return PiiEntry(
            uid=row.get("uid", row.get("id", _safe_int_uid(row, text=text))),
            text=text,
            bio=bio,
            meta={},
        )

    ds = _dataset_from_hf(hf_tag=hf_tag, name=name, from_row=from_row, token=hf_token)
    ds.dump_pickle(cache_path)
    return ds


def load_kiji_da() -> PiiDataset:
    hf_tag = "DataikuNLP/kiji-pii-training-data"
    name = "kiji-pii-da"
    cache_path = Path(PII_EVAL_CACHE_PATH) / f"datasets/{name}.pickle"

    if cache_path.exists():
        return PiiDataset.load_pickle(cache_path)

    def from_row(row: dict[str, Any]) -> PiiEntry | None:
        if row["language"] != "Danish":
            return None
        extra_features = ["coreferences", "country"]
        text = row["text"]
        spans = [
            LabelSpan(
                start=x["start"],
                end=x["end"],
                text=x["value"],
                label=x["label"],
            )
            for x in row["privacy_mask"]
        ]
        return PiiEntry(
            uid=int(hashlib.md5(text.encode()).hexdigest(), 16),
            text=text,
            bio=spans_to_bio(text=text, spans=spans),
            meta={feature: row[feature] for feature in extra_features},
        )

    ds = PiiDataset.from_hf_tag(hf_tag=hf_tag, from_row=from_row, name=name)
    ds.dump_pickle(cache_path)

    return ds


def load_openpii_1m_da() -> PiiDataset:
    hf_tag = "ai4privacy/pii-masking-openpii-1m"
    name = "openpii-1m-da"
    cache_path = Path(PII_EVAL_CACHE_PATH) / f"datasets/{name}.pickle"

    if cache_path.exists():
        return PiiDataset.load_pickle(cache_path)

    def from_row(row: dict[str, Any]) -> PiiEntry | None:
        if row["language"] != "da":
            return None
        uid_feature = "uid"
        text_feature = "source_text"
        spans_feature = "privacy_mask"
        extra_features = ["split", "region", "script"]
        # "mbert_tokens", "mbert_token_classes", not included for simplicity
        text = row[text_feature]
        spans = [
            LabelSpan(
                start=y["start"],
                end=y["end"],
                text=row[text_feature][y["start"] : y["end"]],
                label=y["label"],
            )
            for y in row[spans_feature]
        ]
        return PiiEntry(
            uid=row[uid_feature],
            text=text,
            bio=spans_to_bio(text=text, spans=spans),
            meta={feature: row[feature] for feature in extra_features},
        )

    ds = PiiDataset.from_hf_tag(hf_tag=hf_tag, from_row=from_row, name=name)
    ds.dump_pickle(cache_path)

    return ds


def load_pii_synth_alpha() -> PiiDataset:
    hf_tag = "jharder/pii-synth-alpha"
    name = "pii-synth-alpha"
    cache_path = _cache_path_for(name)

    if cache_path.exists():
        return PiiDataset.load_pickle(cache_path)

    hf_token = _load_hf_token()

    def from_row(row: dict[str, Any]) -> PiiEntry:
        text = str(row.get("text", ""))
        bio = [_labelspan_from_dict(span, text) for span in row.get("bio", [])]
        return PiiEntry(
            uid=row.get("uid", row.get("id", _safe_int_uid(row, text=text))),
            text=text,
            bio=bio,
            meta={},
        )

    ds = _dataset_from_hf(hf_tag=hf_tag, name=name, from_row=from_row, token=hf_token)
    ds.dump_pickle(cache_path)
    return ds


all_datasets: list[Callable[[], PiiDataset]] = [
    load_dane,
    load_dansk_ner,
    load_dimmi,
    load_kiji_da,
    load_openpii_1m_da,
    load_pii_synth_alpha,
]


def load_all_datasets() -> list[PiiDataset]:
    return [load_fct() for load_fct in all_datasets]


if __name__ == "__main__":
    sets = load_all_datasets
