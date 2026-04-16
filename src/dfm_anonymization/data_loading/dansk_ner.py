from datetime import datetime
from pathlib import Path

import pandas as pd

from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.util.spans import LabelSpan, Span, spans_to_bio


def generate_dataset_file(override: bool = False):
    timestamp = datetime.now().isoformat()
    gen_path = Path("data/ground_truth/dansk-ner.json")
    if gen_path.exists() and not override:
        return
    base = "hf://datasets/chcaa/dansk-ner/"
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "dev": "data/dev-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    df_train = pd.read_parquet(base + splits["train"])
    df_dev = pd.read_parquet(base + splits["dev"])
    df_test = pd.read_parquet(base + splits["test"])
    df = pd.concat([df_train, df_dev, df_test])
    df = df.reset_index()

    entries = []
    for idx, row in df.iterrows():
        text = row["text"]
        label_spans = [LabelSpan(**s) for s in row["ents"]]
        token_spans = [Span(start=s["start"], end=s["end"]) for s in row["tokens"]]
        tokens = [text[s.start : s.end] for s in token_spans]
        bio_tags = spans_to_bio(label_spans=label_spans, token_spans=token_spans)

        entries.append(
            PiiDataEntry(
                text=text,
                tokens=tokens,
                id=idx,
                bio_tags=bio_tags,
            )
        )

    dataset = PiiDataset(
        metadata={"dataset_name": "dansk-ner", "timestamp": timestamp}, entries=entries
    )

    dataset.to_file(gen_path)


if __name__ == "__main__":
    generate_dataset_file()
