import hashlib
from pathlib import Path
from typing import Any

from pii_eval.constants import PII_EVAL_CACHE_PATH
from pii_eval.data import (
    PiiDataset,
    PiiEntry,
)
from pii_eval.span import LabelSpan, spans_to_bio


def load_dane() -> PiiDataset:
    hf_tag = "alexandrainst/dane"
    name = "dane"
    raise NotImplementedError


def load_dansk_ner() -> PiiDataset:
    hf_tag = "chcaa/dansk-ner"
    name = "dansk-ner"
    raise NotImplementedError


def load_dimmi() -> PiiDataset:
    hf_tag = "jharder/dimmi"
    name = "dimmi"
    raise NotImplementedError


def load_pii_synth_alpha() -> PiiDataset:
    hf_tag = "jharder/pii-synth-alpha"
    name = "pii-synth-alpha"
    raise NotImplementedError


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


if __name__ == "__main__":
    datasets = []
    for x in [
        load_dane,
        load_dansk_ner,
        load_dimmi,
        load_kiji_da,
        load_openpii_1m_da,
    ]:
        datasets.append(x())

    pass
