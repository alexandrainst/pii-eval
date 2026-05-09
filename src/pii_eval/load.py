import hashlib
from typing import Any

import datasets

from pii_eval.data import (
    PiiDataset,
    PiiEntryLazy,
    FromDatasets,
)
from pii_eval.util import LabelSpan, spans_to_bio


def load_kiji() -> PiiDataset:
    def from_row(row: dict[str, Any]) -> PiiEntryLazy:
        extra_features = ["coreferences", "language", "country"]
        text_f = lambda: row["text"]
        spans_f = lambda: [
            LabelSpan(
                start=x["start"],
                end=x["end"],
                text=x["value"],
                label=x["label"],
            )
            for x in row["privacy_mask"]
        ]
        return PiiEntryLazy(
            uid_f=lambda: int(hashlib.md5(text_f().encode()).hexdigest(), 16),
            text_f=text_f,
            spans_f=spans_f,
            bio_f=lambda: spans_to_bio(text=text_f(), spans=spans_f()),
            extra_f=lambda: {feature: row[feature] for feature in extra_features},
        )

    return FromDatasets.from_hf_tag(
        "DataikuNLP/kiji-pii-training-data", from_row=from_row
    )


def load_openpii_1m() -> PiiDataset:
    def from_row(row: dict[str, Any]) -> PiiEntryLazy:
        uid_feature = "uid"
        text_feature = "source_text"
        spans_feature = "privacy_mask"
        # bio_token_feature = "mbert_tokens"
        # bio_label_feature = "mbert_token_classes"
        text_f = lambda: row[text_feature]
        spans_f = lambda: [
            LabelSpan(
                start=y["start"],
                end=y["end"],
                text=row[text_feature][y["start"] : y["end"]],
                label=y["label"],
            )
            for y in row[spans_feature]
        ]
        return PiiEntryLazy(
            uid_f=lambda: row[uid_feature],
            text_f=text_f,
            spans_f=spans_f,
            bio_f=lambda: spans_to_bio(text=text_f(), spans=spans_f()),
            extra_f=lambda: {
                feature: row[feature]
                for feature in set(row.keys()).difference([uid_feature, text_feature, spans_feature])
            },
        )

    return FromDatasets.from_hf_tag(
        hf_tag="ai4privacy/pii-masking-openpii-1m", from_row=from_row
    )


kiji = load_kiji()
openpii = load_openpii_1m()

pass
