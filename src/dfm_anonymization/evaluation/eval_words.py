from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dfm_anonymization.inference.dataset import PiiDataset
from dfm_anonymization.util.misc import DATASET_LABEL_MAP, MODEL_LABEL_MAP
from dfm_anonymization.util.util import to_dict_recursive


def map_bio_tags(tags: List[str], label_map: Dict[str, str | None]):
    sentinel = object()
    mapped_tags: List[str] = []
    append = mapped_tags.append

    for tag in tags:
        if tag == "O" or "-" not in tag:
            append("O" if tag == "" else tag)
            continue

        prefix, label = tag.split("-", 1)
        mapped_label = label_map.get(label, sentinel)

        if mapped_label is sentinel:
            append(tag)
        elif mapped_label is None:
            append("O")
        elif mapped_label == label:
            append(tag)
        else:
            append(f"{prefix}-{mapped_label}")

    return mapped_tags


@dataclass
class EvaluationResult:
    precision: float
    f1: float
    f5: float
    recall: float


def evaluate(
    ground_truth: PiiDataset, predictions: PiiDataset
) -> EvaluationResult:
    truth_entries = {entry.id: entry for entry in ground_truth}
    prediction_entries = {entry.id: entry for entry in predictions}

    missing_predictions = truth_entries.keys() - prediction_entries.keys()
    extra_predictions = prediction_entries.keys() - truth_entries.keys()
    if missing_predictions or extra_predictions:
        raise ValueError(
            "Mismatching entry ids between ground truth and predictions: "
            f"missing_predictions={sorted(missing_predictions)}, "
            f"extra_predictions={sorted(extra_predictions)}"
        )

    true_positive = false_positive = false_negative = 0

    for entry_id, truth_entry in truth_entries.items():
        prediction_entry = prediction_entries[entry_id]
        truth_tags = map_bio_tags(
            truth_entry.bio_tags or [], DATASET_LABEL_MAP[ground_truth.name])
        predicted_tags = map_bio_tags(
            prediction_entry.bio_tags or [], MODEL_LABEL_MAP[predictions.name])

        if len(truth_tags) != len(predicted_tags):
            raise ValueError(
                f"Mismatching tag counts for entry {entry_id}: "
                f"truth={len(truth_tags)}, predicted={len(predicted_tags)}"
            )

        for truth_tag, predicted_tag in zip(truth_tags, predicted_tags):
            truth_is_entity = truth_tag != "O"
            predicted_is_entity = predicted_tag != "O"

            if predicted_tag == truth_tag and truth_is_entity:
                true_positive += 1
                continue
            if predicted_is_entity:
                false_positive += 1
            if truth_is_entity:
                false_negative += 1

    def _safe_divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    denominator_f1 = precision + recall
    f1 = (2 * precision * recall / denominator_f1) if denominator_f1 else 0.0
    beta_sq = 25
    denom_f5 = (beta_sq * precision) + recall
    f5 = ((1 + beta_sq) * precision * recall / denom_f5) if denom_f5 else 0.0

    return EvaluationResult(
        precision=precision,
        f1=f1,
        f5=f5,
        recall=recall,
    )


def evaluate_all():
    results = []
    for predictions_path in Path("data/infer_ds").rglob("*.json"):
        dataset_name = predictions_path.stem
        ground_truth_path = f"data/ground_truth/{dataset_name}.json"
        predictions = PiiDataset.from_file(predictions_path)
        ground_truth = PiiDataset.from_file(ground_truth_path)
        results.append(
            evaluate(ground_truth=ground_truth, predictions=predictions))
    df = pd.Dataframe(to_dict_recursive(results))
    print(df)
    pass


if __name__ == "__main__":
    evaluate_all()
