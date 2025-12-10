#!/usr/bin/env python3
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

from dfm_anonymization.util.misc import MODEL_LABEL_MAP, DATASET_LABEL_MAP


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
    whitelist_model: Optional[Dict[str, Optional[str]]] = None,
    whitelist_dataset: Optional[Dict[str, Optional[str]]] = None
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
        raise ValueError(
            ", ".join(
                msg for msg in (
                    missing_msg,
                    extra_msg) if msg))

    tp = fp = fn = 0
    total_chars = 0
    for doc_id in truth_ids:
        truth_entry = truths[doc_id]
        pred_entry = predictions[doc_id]
        text = truth_entry["text"]
        total_chars += len(text)

        pred_text = pred_entry["text"]
        if len(pred_text) != len(text):
            raise ValueError(
                f"Text length mismatch for doc {doc_id}: "
                f"truth={len(text)} pred={len(pred_text)}"
            )

        truth_mask = mask_from_spans(
            len(text), truth_entry.get("spans", []), whitelist_dataset)
        pred_mask = mask_from_spans(
            len(text), pred_entry.get("spans", []), whitelist_model)
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
    timestamp = datetime.datetime.now().isoformat()
    out_path = Path(f"data/results/{timestamp}.json")
    results = []
    for dataset_name in DATASET_LABEL_MAP.keys():
        truth_path = Path(f"data/span/{dataset_name}.json")
        truths = load_records(path=truth_path)
        predictions_base = Path(f"data/infer/{dataset_name}")
        for predictions_path in predictions_base.glob("*.json"):
            model_name = predictions_path.stem
            model_label_map = MODEL_LABEL_MAP[model_name] \
                if model_name in MODEL_LABEL_MAP else None
            dataset_label_map = DATASET_LABEL_MAP[dataset_name]
            whitelist_model = [
                k for k, v in model_label_map.items()
                if v and v in dataset_label_map.values()
            ] if model_label_map else None
            whitelist_ds = [
                k for k, v in dataset_label_map.items()
                if v and (not model_label_map or v in model_label_map.values())
            ]
            predictions = load_records(path=predictions_path)
            tp, fp, fn, tn = evaluate_documents(
                predictions, truths, whitelist_model, whitelist_ds)

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  ) if (precision + recall) else 0.0
            beta_sq = 25
            f5 = ((1 + beta_sq) * precision * recall /
                  (beta_sq * precision + recall)) if (beta_sq * precision + recall) else 0.0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

            # if model_label_map:
            #     print(f"model labels: {[k for k, v in model_label_map.items() if v]}")
            # print(f"dataset labels: {[k for k, v in dataset_label_map.items() if v]}")
            # print(f"whitelist model: {whitelist_model}")
            # print(f"whitelist dataset: {whitelist_ds}")
            # print(
            #     "\n".join(
            #         [
            #             f"True Positives : {tp}",
            #             f"False Positives: {fp}",
            #             f"False Negatives: {fn}",
            #             f"True Negatives: {tn}",
            #             f"Precision     : {precision:.4f}",
            #             f"Recall        : {recall:.4f}",
            #             f"F1 Score      : {f1:.4f}",
            #             f"F5 Score      : {f5:.4f}",
            #             f"Accuracy      : {accuracy:.4f}",
            #         ]
            #     )
            # )

            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "labels_dataset": whitelist_ds,
                "labels_model": whitelist_model,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f5": f5,
                "accuracy": accuracy
            })

            out_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
