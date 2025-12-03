import json
from pathlib import Path
from collections.abc import Iterable as ABCIterable
from typing import Any, Dict, Iterable, List

DATASET_NAME = "mattmdjaga/text-anonymization-benchmark-train"


def _parse_raw_spans(raw_spans: Any) -> list[Any]:
    if raw_spans is None:
        return []
    if isinstance(raw_spans, str):
        try:
            parsed = json.loads(raw_spans)
        except json.JSONDecodeError:
            return []
        else:
            return parsed or []
    if isinstance(raw_spans, dict):
        return [raw_spans]
    if isinstance(raw_spans, ABCIterable):
        return list(raw_spans)
    return []


def _iter_span_dicts(raw: Any) -> Iterable[Dict[str, Any]]:
    stack = list(_parse_raw_spans(raw))
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except json.JSONDecodeError:
                continue
        if current is None:
            continue
        if isinstance(current, dict):
            if (
                "start" in current
                or "start_offset" in current
                or "startIndex" in current
            ):
                yield current
            elif "entity_mentions" in current and isinstance(
                current["entity_mentions"], ABCIterable
            ):
                stack.extend(current["entity_mentions"])
            else:
                stack.extend(v for v in current.values())
        elif isinstance(current, ABCIterable):
            stack.extend(current)


def _extract_spans(
    raw_spans: Any, full_text: str | None = None
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    seen = set()
    for span in _iter_span_dicts(raw_spans):
        label = (
            span.get("label")
            or span.get("entity_type")
            or span.get("entityLabel")
            or span.get("type")
        )
        start = (
            span.get("start")
            or span.get("start_position")
            or span.get("startIndex")
            or span.get("start_offset")
        )
        end = (
            span.get("end")
            or span.get("end_position")
            or span.get("endIndex")
            or span.get("end_offset")
        )
        if label is None or start is None or end is None:
            continue
        try:
            start_idx = int(start)
            end_idx = int(end)
        except (TypeError, ValueError):
            continue
        key = (label, start_idx, end_idx)
        if key in seen:
            continue
        seen.add(key)
        spans.append({"label": label, "start": start_idx, "end": end_idx})
    spans.sort(key=lambda item: (item["start"], item["end"]))
    if isinstance(full_text, str):
        text_len = len(full_text)
        for span in spans:
            start_idx = span["start"]
            end_idx = span["end"]
            if start_idx < 0:
                start_idx = 0
            elif start_idx > text_len:
                start_idx = text_len
            if end_idx < start_idx:
                end_idx = start_idx
            elif end_idx > text_len:
                end_idx = text_len
            span["text"] = full_text[start_idx:end_idx]
    return spans


def main() -> None:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME)["train"]

    raw_fp = Path(
        "data/raw/mattmdjaga/text-anonymization-benchmark-train.json"
    )
    raw_fp.resolve()
    if not raw_fp.exists():
        raw_fp.parent.mkdir(exist_ok=True, parents=True)
        with open(raw_fp, "w") as f:
            json.dump(dataset.to_dict(), f, indent=True)

    output = []

    for idx, example in enumerate(dataset):
        full_text = example.get("full_text") or example.get("text")
        if full_text is None:
            raise ValueError("Example missing full text content")

        span_source = (
            example.get("spans")
            or example.get("entities")
            or example.get("annotations")
            or []
        )

        output.append(
            {
                "document": example.get("document")
                or example.get("doc_id")
                or example.get("id")
                or idx,
                "full_text": full_text,
                "spans": _extract_spans(span_source, full_text),
            }
        )

    out_fp = Path("data/processed/spans/mattmdjaga.json").resolve()
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    with out_fp.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=True)

    reduced_fp = Path(
        "data/processed/spans/reduced/mattmdjaga_red.json").resolve()
    reduced_fp.parent.mkdir(parents=True, exist_ok=True)

    with reduced_fp.open("w", encoding="utf-8") as f:
        json.dump(output[:5], f, ensure_ascii=False, indent=True)


if __name__ == "__main__":
    main()
