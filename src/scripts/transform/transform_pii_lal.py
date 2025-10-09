import json
from pathlib import Path
from typing import List

data_fp = Path(
    "data/raw/download_cache/kagglehub/competitions/"
    "pii-detection-removal-from-educational-data"
).resolve()

with open(data_fp / "train.json", "r") as f:
    documents: List[dict] = json.load(f)

classes = {
    "B-EMAIL",
    "B-ID_NUM",
    "B-NAME_STUDENT",
    "B-PHONE_NUM",
    "B-STREET_ADDRESS",
    "B-URL_PERSONAL",
    "B-USERNAME",
    "I-ID_NUM",
    "I-NAME_STUDENT",
    "I-PHONE_NUM",
    "I-STREET_ADDRESS",
    "I-URL_PERSONAL",
    "O",
}

documents_with_spans = []

for document in documents:
    assert set(document.keys()) == {
        "document",
        "full_text",
        "tokens",
        "trailing_whitespace",
        "labels",
    }
    tokens = document["tokens"]
    labels = document["labels"]
    trailing_whitespace = document["trailing_whitespace"]
    assert len(tokens) == len(labels) == len(trailing_whitespace)
    for token, label in zip(tokens, labels):
        assert label in classes

    document_id = document["document"]
    char_position = 0
    active_span = None
    spans = []
    text_parts = []

    for token, label, has_whitespace in zip(tokens, labels, trailing_whitespace):
        token_start = char_position
        token_end = token_start + len(token)

        if label == "O":
            if active_span is not None:
                spans.append(active_span)
                active_span = None
        else:
            prefix, entity_label = label.split("-", 1)
            if prefix == "B" or active_span is None:
                if active_span is not None:
                    spans.append(active_span)
                active_span = {
                    "label": entity_label,
                    "start": token_start,
                    "end": token_end,
                }
            elif prefix == "I":
                if active_span["label"] != entity_label:
                    spans.append(active_span)
                    active_span = {
                        "label": entity_label,
                        "start": token_start,
                        "end": token_end,
                    }
                else:
                    active_span["end"] = token_end
            else:
                raise ValueError(f"Unexpected prefix: {prefix}")

        char_position = token_end
        text_parts.append(token)

        if has_whitespace:
            char_position += 1
            text_parts.append(" ")

    if active_span is not None:
        spans.append(active_span)

    reconstructed_text = "".join(text_parts)
    if reconstructed_text != document["full_text"]:
        raise ValueError(f"Full text mismatch for document {document_id}")

    text_len = len(reconstructed_text)
    for span in spans:
        start_idx = max(0, min(span["start"], text_len))
        end_idx = max(start_idx, min(span["end"], text_len))
        span["start"] = start_idx
        span["end"] = end_idx
        span["text"] = reconstructed_text[start_idx:end_idx]

    documents_with_spans.append(
        {
            "document": document_id,
            "full_text": reconstructed_text,
            "spans": spans,
        }
    )


out_fp = Path("data/raw/transformed/pii_lal.json").resolve()

with open(out_fp, "w") as f:
    json.dump(documents_with_spans, f, ensure_ascii=False, indent=True)

# Save reduced version with first 5 entries including span text
reduced_path = Path("data/raw/transformed/reduced/pii_lal_red.json").resolve()
reduced_path.parent.mkdir(parents=True, exist_ok=True)
with open(reduced_path, "w") as handle:
    json.dump(documents_with_spans[:5], handle, ensure_ascii=False, indent=True)

print()
