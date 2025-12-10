#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

DEFAULT_DATASET = "dansk_ner"
DEFAULT_MODEL = "scandiner"
DEFAULT_REDUCED = "" # "/reduced"
DEFAULT_PII_LABELS = [
    "CREDITCARD",
    "CREDIT_CARD",
    "EMAIL",
    "NAME",
    "NAME_STUDENT",
    "ORGANIZATION",
    "PER",
    "PERSON",
    "PHONE_NUM",
    "PHONE_NUMBER",
    "STREET_ADDRESS",
    "USERNAME",
    "URL_PERSONAL"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Character-level, label-agnostic span evaluation."
    )
    parser.add_argument(
        "--predictions",
        "-p",
        type=Path,
        default=Path(f"data/infer/{DEFAULT_DATASET}{DEFAULT_REDUCED}/{DEFAULT_MODEL}.json"),
        help="Path to the predictions JSON file.",
    )
    parser.add_argument(
        "--truth",
        "-t",
        type=Path,
        default=Path(f"data/span{DEFAULT_REDUCED}/{DEFAULT_DATASET}.json"),
        help="Path to the ground truth JSON file.",
    )
    return parser.parse_args()


def load_records(path: Path) -> Dict[int, dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    records: Dict[int, dict] = {}
    for entry in data:
        doc_id = entry.get("document")
        if doc_id is None:
            raise ValueError(f"Entry missing 'document' field in {path}")
        if doc_id in records:
            raise ValueError(f"Duplicate document id {doc_id} in {path}")
        records[doc_id] = entry
    return records


def mask_from_spans(
    length: int, spans: Iterable[dict], whitelist: Optional[Set[str]] = None
) -> bytearray:
    mask = bytearray(length)
    for span in spans:
        label = span.get("label")
        if whitelist is not None and label not in whitelist:
            continue
        start = int(span["start"])
        end = int(span["end"])
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"Invalid span bounds: {start} -> {end}")
        if end > length + 1:
            raise ValueError(f"Span end {end} exceeds text length {length}")
        if start == end:
            continue
        mask[start:end] = b"\x01" * (end - start)
    return mask


def evaluate_documents(
    predictions: Dict[int, dict],
    truths: Dict[int, dict],
    whitelist: Optional[Set[str]] = None,
) -> Tuple[int, int, int, int]:
    pred_ids = set(predictions)
    truth_ids = set(truths)
    missing_preds = truth_ids - pred_ids
    extra_preds = pred_ids - truth_ids
    if missing_preds or extra_preds:
        missing_msg = (
            f"Missing predictions for documents: {sorted(missing_preds)}"
            if missing_preds
            else ""
        )
        extra_msg = (
            f"Ground truth missing for documents: {sorted(extra_preds)}"
            if extra_preds
            else ""
        )
        raise ValueError(", ".join(msg for msg in (missing_msg, extra_msg) if msg))

    tp = fp = fn = 0
    total_chars = 0
    for doc_id in truth_ids:
        truth_entry = truths[doc_id]
        pred_entry = predictions[doc_id]
        text = truth_entry["full_text"]
        total_chars += len(text)

        pred_text = pred_entry["full_text"]
        if len(pred_text) != len(text):
            raise ValueError(
                f"Text length mismatch for doc {doc_id}: "
                f"truth={len(text)} pred={len(pred_text)}"
            )

        truth_mask = mask_from_spans(
            len(text), truth_entry.get("spans", []), whitelist
        )
        pred_mask = mask_from_spans(len(text), pred_entry.get("spans", []), whitelist)
        for pred_char, truth_char in zip(pred_mask, truth_mask):
            if pred_char:
                if truth_char:
                    tp += 1
                else:
                    fp += 1
            elif truth_char:
                fn += 1
    tn = total_chars - (tp + fp + fn)
    return tp, fp, fn, tn


def main() -> None:
    args = parse_args()
    whitelist = DEFAULT_PII_LABELS
    predictions = load_records(path=args.predictions)
    truths = load_records(path=args.truth)
    tp, fp, fn, tn = evaluate_documents(predictions, truths, whitelist)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    print(
        "\n".join(
            [
                f"True Positives : {tp}",
                f"False Positives: {fp}",
                f"False Negatives: {fn}",
                f"True Negatives: {tn}",
                f"Precision     : {precision:.4f}",
                f"Recall        : {recall:.4f}",
                f"F1 Score      : {f1:.4f}",
                f"Accuracy      : {accuracy:.4f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
