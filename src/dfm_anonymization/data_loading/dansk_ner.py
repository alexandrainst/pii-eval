from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd

from dfm_anonymization.dataset import PiiDataEntry, PiiDataset


@dataclass
class Span:
    start: int
    end: int

@dataclass
class LabelSpan(Span):
    label: str


def spans_to_bio(
        label_spans: List[LabelSpan], token_spans: List[Span]) -> List[str]:
    if not token_spans:
        return []

    sorted_labels = sorted(label_spans, key=lambda span: span.start)
    tags: List[str] = []
    label_idx = 0
    current = sorted_labels[label_idx] if sorted_labels else None

    for token in token_spans:
        while current is not None and token.start >= current.end:
            label_idx += 1
            if label_idx >= len(sorted_labels):
                current = None
                break
            current = sorted_labels[label_idx]

        if current is None or token.end <= current.start:
            tags.append("O")
            continue

        prefix = "B" if token.start <= current.start else "I"
        tags.append(f"{prefix}-{current.label}")

    return tags


def generate_dataset_file(override: bool = False):
    timestamp = datetime.now().isoformat()
    gen_path = Path("data/ground_truth/dansk-ner.json")
    if gen_path.exists() and not override:
        return
    base = "hf://datasets/chcaa/dansk-ner/"
    splits = {'train': 'data/train-00000-of-00001.parquet',
              'dev': 'data/dev-00000-of-00001.parquet',
              'test': 'data/test-00000-of-00001.parquet'}
    df_train = pd.read_parquet(base + splits["train"])
    df_dev = pd.read_parquet(base + splits["dev"])
    df_test = pd.read_parquet(base + splits["test"])
    df = pd.concat([df_train, df_dev, df_test])
    df = df.reset_index()

    entries = []
    for idx, row in df.iterrows():
        text = row["text"]
        label_spans = [LabelSpan(**s) for s in row["ents"]]
        token_spans = [
            Span(start=s['start'], end=s['end']) for s in row["tokens"]]
        tokens = [text[s.start:s.end] for s in token_spans]
        bio_tags = spans_to_bio(label_spans=label_spans, token_spans=token_spans)

        entries.append(PiiDataEntry(
            text=text,
            tokens=tokens,
            id=idx,
            bio_tags=bio_tags,
        ))

    dataset = PiiDataset(
        metadata={
            "dataset_name": "dansk-ner",
            "timestamp": timestamp
        },
        entries=entries
    )

    dataset.to_file(gen_path)


if __name__ == "__main__":
    generate_dataset_file()
