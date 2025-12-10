import json
from pathlib import Path
from typing import List

input_fp = Path(
    "data/raw/presidio/generated_size_1500_date_October_10_2025.json"
).resolve()

with open(input_fp, "r") as f:
    documents: List[dict] = json.load(f)

transformed = []

for idx, document in enumerate(documents):
    assert set(document.keys()) >= {"full_text", "spans"}

    spans = []
    text = document["full_text"]
    text_len = len(text)
    for span in document["spans"]:
        start = int(span["start_position"])
        end = int(span["end_position"])
        if start < 0:
            start = 0
        elif start > text_len:
            start = text_len
        if end < start:
            end = start
        elif end > text_len:
            end = text_len
        spans.append(
            {
                "label": span["entity_type"],
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )

    transformed.append(
        {
            "document": idx,
            "full_text": document["full_text"],
            "spans": spans,
        }
    )


out_fp = Path("data/processed/span/presidio_gen.json").resolve()

with open(out_fp, "w") as f:
    json.dump(transformed, f, indent=True)

reduced_fp = Path("data/processed/reduced/presidio_gen_red.json").resolve()
reduced_fp.parent.mkdir(parents=True, exist_ok=True)
with open(reduced_fp, "w") as f:
    json.dump(transformed[:5], f, ensure_ascii=False)
