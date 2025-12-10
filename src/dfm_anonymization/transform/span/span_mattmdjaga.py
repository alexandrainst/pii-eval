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


def _resolve_majority_label(spans: List[Dict[str, Any]]) -> str:
    label_segments: Dict[str, List[tuple[int, int]]] = {}
    for span in spans:
        label_segments.setdefault(span["label"], []).append(
            (span["start"], span["end"])
        )
    label_scores: Dict[str, int] = {}
    for label, segments in label_segments.items():
        segments.sort()
        coverage = 0
        current_start, current_end = segments[0]
        for start, end in segments[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                coverage += current_end - current_start
                current_start, current_end = start, end
        coverage += current_end - current_start
        label_scores[label] = coverage
    max_score = max(label_scores.values())
    candidates = [
        label for label, score in label_scores.items() if score == max_score
    ]
    if len(candidates) == 1:
        return candidates[0]
    for span in spans:
        if span["label"] in candidates:
            return span["label"]
    return spans[0]["label"]


def _merge_overlapping_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not spans:
        return []
    merged: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(spans):
        current_group = [spans[idx]]
        current_start = spans[idx]["start"]
        current_end = spans[idx]["end"]
        idx += 1
        while idx < len(spans) and spans[idx]["start"] < current_end:
            current_group.append(spans[idx])
            current_end = max(current_end, spans[idx]["end"])
            idx += 1
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            merged.append(
                {
                    "label": _resolve_majority_label(current_group),
                    "start": current_start,
                    "end": current_end,
                }
            )
    return merged


def _extract_spans(
    raw_spans: Any, text: str | None = None
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
        if end_idx - start_idx <= 5:
            continue
        key = (label, start_idx, end_idx)
        if key in seen:
            continue
        seen.add(key)
        spans.append({"label": label, "start": start_idx, "end": end_idx})
    spans.sort(key=lambda item: (item["start"], item["end"]))
    spans = _merge_overlapping_spans(spans)
    if isinstance(text, str):
        text_len = len(text)
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
            span["text"] = text[start_idx:end_idx]
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
        text = example.get("text") or example.get("text")
        if text is None:
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
                "text": text,
                "spans": _extract_spans(span_source, text),
            }
        )

    out_fp = Path("data/processed/span/mattmdjaga.json").resolve()
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    with out_fp.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=True)

    reduced_fp = Path(
        "data/processed/span/reduced/mattmdjaga.json").resolve()
    reduced_fp.parent.mkdir(parents=True, exist_ok=True)

    with reduced_fp.open("w", encoding="utf-8") as f:
        json.dump(output[:5], f, ensure_ascii=False, indent=True)


if __name__ == "__main__":
    main()
