import json
from pathlib import Path

import pandas as pd

# GPE = Geopolitical entities

base = "hf://datasets/chcaa/dansk-ner/"
splits = {'train': 'data/train-00000-of-00001.parquet',
          'dev': 'data/dev-00000-of-00001.parquet',
          'test': 'data/test-00000-of-00001.parquet'}
df_train = pd.read_parquet(base + splits["train"])
df_dev = pd.read_parquet(base + splits["dev"])
df_test = pd.read_parquet(base + splits["test"])
df = pd.concat([df_train, df_dev, df_test])
df = df.reset_index()

# DONE: I need to filter the dataframe so that only rows where at column 'ents'
# the list contains a dict with 'label' = 'PERSON'
# person_mask = df["ents"].map(lambda ents: any(
#     ent.get("label") == "PERSON" for ent in ents))
# df = df[person_mask].reset_index(drop=True)

output_path = Path("data/processed/span/da/dansk_ner.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

if df.empty:
    output_path.write_text("[]", encoding="utf-8")
    raise SystemExit(0)

text_column_candidates = ("full_text", "text", "content", "raw_text")
text_column = next(
    (col for col in text_column_candidates if col in df.columns),
    None)
if text_column is None:
    for column in df.columns:
        if column == "ents":
            continue
        value = df.iloc[0][column]
        if isinstance(value, str):
            text_column = column
            break
if text_column is None:
    raise ValueError("Unable to identify a text column to export.")

records = []
for row in df.itertuples():
    full_text = getattr(row, text_column)
    if not isinstance(full_text, str):
        full_text = "" if full_text is None else str(full_text)

    spans = []
    for ent in getattr(row, "ents"):
        start = ent.get("start")
        end = ent.get("end")
        if start is not None:
            start = int(start)
        if end is not None:
            end = int(end)
        span_text = ent.get("text")
        if span_text is None and start is not None and end is not None:
            span_text = full_text[start:end]
        spans.append(
            {
                "label": ent.get("label"),
                "start": start,
                "end": end,
                "text": span_text,
            }
        )

    if not spans:
        continue

    document_id = row.Index if hasattr(row, "Index") else len(records)
    try:
        document_id = int(document_id)
    except (TypeError, ValueError):
        document_id = str(document_id)

    records.append(
        {
            "document": document_id,
            "full_text": full_text,
            "spans": spans,
        }
    )

red_output_path = output_path.parent / "reduced" / output_path.name
red_output_path.write_text(json.dumps(
    records[:5], ensure_ascii=False, indent=4))
output_path.write_text(json.dumps(
    records, ensure_ascii=False, indent=4))
