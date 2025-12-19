from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dfm_anonymization.inference.dataset import PiiDataset
from dfm_anonymization.util.misc import (
    DATASET_LABEL_MAP, LABEL_DESCRIPTIONS_DA, MODEL_LABEL_MAP)
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
    dataset: str
    model: str
    label: str
    precision: float
    f1: float
    f5: float
    recall: float


def evaluate(
    ground_truth: PiiDataset, predictions: PiiDataset
) -> List[EvaluationResult]:
    dataset_name = ground_truth.metadata["dataset_name"]
    model_name = predictions.metadata["model_name"]
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
    label_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    truth_labels: set[str] = set()
    predicted_labels: set[str] = set()

    for entry_id, truth_entry in truth_entries.items():
        prediction_entry = prediction_entries[entry_id]
        truth_tags = map_bio_tags(
            truth_entry.bio_tags or [], DATASET_LABEL_MAP[dataset_name])
        predicted_tags = map_bio_tags(
            prediction_entry.bio_tags or [],
            MODEL_LABEL_MAP[model_name] if model_name in MODEL_LABEL_MAP
            else LABEL_DESCRIPTIONS_DA
        )

        if len(truth_tags) != len(predicted_tags):
            raise ValueError(
                f"Mismatching tag counts for entry {entry_id}: "
                f"truth={len(truth_tags)}, predicted={len(predicted_tags)}"
            )

        for truth_tag, predicted_tag in zip(truth_tags, predicted_tags):
            truth_is_entity = truth_tag != "O"
            predicted_is_entity = predicted_tag != "O"
            truth_label = truth_tag[2:] if truth_is_entity else None
            predicted_label = predicted_tag[2:] if predicted_is_entity else None

            if truth_label:
                truth_labels.add(truth_label)
            if predicted_label:
                predicted_labels.add(predicted_label)

            if truth_is_entity and predicted_is_entity:
                if predicted_label == truth_label:
                    label_counts[truth_label][0] += 1
                else:
                    label_counts[predicted_label][1] += 1
                    label_counts[truth_label][2] += 1
            elif predicted_is_entity:
                label_counts[predicted_label][1] += 1
            elif truth_is_entity:
                label_counts[truth_label][2] += 1

    def _safe_divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    beta_sq = 25
    intersection_labels = truth_labels & predicted_labels
    results: List[EvaluationResult] = []
    append_result = results.append

    for label in sorted(intersection_labels):
        true_positive, false_positive, false_negative = label_counts.get(
            label, (0, 0, 0)
        )
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        denominator_f1 = precision + recall
        f1 = (2 * precision * recall / denominator_f1) if denominator_f1 else 0.0
        denom_f5 = (beta_sq * precision) + recall
        f5 = (
            (1 + beta_sq) * precision * recall / denom_f5
            if denom_f5
            else 0.0
        )

        append_result(EvaluationResult(
            dataset=dataset_name,
            model=model_name,
            label=label,
            precision=precision,
            f1=f1,
            f5=f5,
            recall=recall,
        ))

    return results


def evaluate_all():
    results = []
    for predictions_path in Path("data/infer_ds").rglob("*.json"):
        dataset_name = predictions_path.stem
        ground_truth_path = f"data/ground_truth/{dataset_name}.json"
        predictions = PiiDataset.from_file(predictions_path)
        ground_truth = PiiDataset.from_file(ground_truth_path)
        results += evaluate(ground_truth=ground_truth, predictions=predictions)
    df = pd.DataFrame(to_dict_recursive(results))
    metrics = ["precision", "f1", "f5", "recall"]
    df = df[["dataset", "model", "label"] + metrics]
    df = df.sort_values(["dataset", "label", "f5"],
                        ascending=[True, True, False])
    best_models = df.groupby("model")[metrics].mean(
    ).sort_values("f5", ascending=False)
    dataset_scores = df.groupby(
        "dataset")[metrics].mean().sort_values("f5", ascending=False)
    best_models_by_label = df.groupby(["label", "model"])[metrics].mean() \
        .reset_index().sort_values(["label", "f5"], ascending=[True, False])
    best_model_by_label = best_models_by_label.loc[
        best_models_by_label.groupby("label").idxmax()["f5"]]
    best_models_by_label = best_models_by_label.set_index(["label", "model"])
    best_model_by_label = best_model_by_label.set_index(["label", "model"])
    print(df.set_index(['dataset', 'model', 'label']), end="\n\n")
    print(best_models_by_label, end="\n\n")
    print(dataset_scores, end="\n\n")
    print(best_models, end="\n\n")
    print(best_model_by_label)
    pass


if __name__ == "__main__":
    evaluate_all()
