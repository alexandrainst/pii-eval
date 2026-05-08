from typing import Any

import datasets

from dfm_anonymization.data_loading.new_structure.data import (
    PiiDataset,
    PiiEntryLazy,
    SpansDatasets,
)
from dfm_anonymization.data_loading.new_structure.util import LabelSpan, spans_to_bio


def load_kiji() -> PiiDataset:
    hf_tag = "DataikuNLP/kiji-pii-training-data"

    def from_row(row: dict[str, Any]) -> PiiEntryLazy:
        spans_feature = "privacy_mask"
        extra_features = ["coreferences", "language", "country"]
        text_f = lambda: row["text"]
        spans_f = lambda: [
            LabelSpan(
                start=x["start"],
                end=x["end"],
                text=x["value"],
                label=x["label"],
            )
            for x in row[spans_feature]
        ]
        return PiiEntryLazy(
            text_f=text_f,
            spans_f=spans_f,
            bio_f=lambda: spans_to_bio(text=text_f(), spans=spans_f()),
            extra_f=lambda: {feature: row[feature] for feature in extra_features},
        )

    return SpansDatasets(
        name=hf_tag.split("/")[-1],
        load_dataset=lambda: datasets.load_dataset(hf_tag)["test"],
        from_row=from_row,
    )


ds = load_kiji()

ds[0].bio

pass
