import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

import pandas as pd

from dfm_anonymization.constants import (
    ALL_ISO_LABELS,
    DATASET_LABEL_MAP,
    MODEL_LABEL_MAP,
)
from dfm_anonymization.dataset import PiiDataset
from dfm_anonymization.util import to_dict_recursive


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
class ConfusionResult:
    tp: int
    fp: int
    fn: int
    tn: int

    @staticmethod
    def zero() -> "ConfusionResult":
        return ConfusionResult(0, 0, 0, 0)

    def rp(self) -> int:
        return self.tp + self.fn

    def pp(self) -> int:
        return self.tp + self.fp

    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


@dataclass
class MetricResult(ConfusionResult):
    precision: float
    f1: float
    f5: float
    recall: float

    @staticmethod
    def from_confusion(c: ConfusionResult) -> "MetricResult":
        precision = c.tp / c.pp() if c.pp() else 0.0
        recall = c.tp / c.rp() if c.rp() else 0.0

        f1_denominator = (2 * c.tp) + c.fp + c.fn
        if f1_denominator > 0:
            f1 = (2.0 * c.tp) / f1_denominator
        else:
            f1 = 0.0

        beta_squared = 25.0
        f5_denominator = ((1.0 + beta_squared) * c.tp) + (beta_squared * c.fn) + c.fp
        if f5_denominator > 0:
            f5 = ((1.0 + beta_squared) * c.tp) / f5_denominator
        else:
            f5 = 0.0

        return MetricResult(
            **asdict(c), precision=precision, f1=f1, f5=f5, recall=recall
        )


@dataclass
class EvaluationResult(MetricResult):
    dataset: str
    model: str
    label: str


def evaluate(
    ground_truth: PiiDataset, predictions: PiiDataset
) -> Dict[str, MetricResult]:
    dataset_name = ground_truth.metadata["dataset_name"]
    model_name = predictions.metadata["model_name"]
    dataset_label_map = (
        DATASET_LABEL_MAP[dataset_name]
        if dataset_name in DATASET_LABEL_MAP
        else {k: k for k in ALL_ISO_LABELS}
    )
    model_label_map = (
        MODEL_LABEL_MAP[model_name]
        if model_name in MODEL_LABEL_MAP
        else {k: k for k in ALL_ISO_LABELS}
    )
    common_iso_labels = set(dataset_label_map.values()).intersection(
        set(model_label_map.values())
    )

    confusion_counts = {}
    for label in common_iso_labels:
        confusion = ConfusionResult.zero()
        confusion_counts[label] = confusion
        for true_entry, pred_entry in zip(ground_truth, predictions):
            for true_tag, pred_tag in zip(true_entry.bio_tags, pred_entry.bio_tags):
                true_orig_label = true_tag[2:] if true_tag != "O" else None
                pred_orig_label = pred_tag[2:] if pred_tag != "O" else None
                true_iso_label = (
                    dataset_label_map[true_orig_label]
                    if true_orig_label in dataset_label_map
                    else None
                )
                pred_iso_label = (
                    model_label_map[pred_orig_label]
                    if pred_orig_label in model_label_map
                    else None
                )
                if true_iso_label and true_iso_label == label:
                    if pred_iso_label and pred_iso_label == label:
                        confusion.tp += 1
                    else:
                        confusion.fn += 1
                else:
                    if pred_iso_label and pred_iso_label == label:
                        confusion.fp += 1
                    else:
                        confusion.tn += 1

    return {
        label: MetricResult.from_confusion(confusion)
        for label, confusion in confusion_counts.items()
    }


def evaluate_all():
    results = []
    performances = {}
    for predictions_path in Path("data/predictions").glob("*/*.json"):
        dataset_name = predictions_path.stem
        model_name = predictions_path.parent.name
        ground_truth_path = f"data/ground_truth/{dataset_name}.json"
        predictions = PiiDataset.from_file(predictions_path)
        ground_truth = PiiDataset.from_file(ground_truth_path)
        metrics_dict = evaluate(ground_truth=ground_truth, predictions=predictions)
        results += [
            EvaluationResult(
                **asdict(metrics), dataset=dataset_name, model=model_name, label=label
            )
            for label, metrics in metrics_dict.items()
        ]
        if model_name not in performances:
            performances[model_name] = []
        performances[model_name].append(
            predictions.metadata["computation_time_per_entry"]
        )
    performances = pd.DataFrame(
        [1.0 / mean(vs) for vs in performances.values()],
        index=performances.keys(),
        columns=["samples/s"],
    )
    df = pd.DataFrame(to_dict_recursive(results))
    timestamp = datetime.now().isoformat()
    metrics = ["precision", "f1", "f5", "recall"]
    df = df[["dataset", "model", "label"] + metrics]
    df = df.sort_values(["dataset", "label", "f5"], ascending=[True, True, False])
    out_path = Path(f"data/results/{timestamp}.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(f"data/results/{timestamp}.json")
    df = df.dropna(how="any")
    best_models = df.groupby("model")[metrics].mean().sort_values("f5", ascending=False)
    dataset_scores = (
        df.groupby("dataset")[metrics].mean().sort_values("f5", ascending=False)
    )
    best_models_by_label = (
        df.groupby(["label", "model"])[metrics]
        .mean()
        .reset_index()
        .sort_values(["label", "f5"], ascending=[True, False])
    )
    best_model_by_label = best_models_by_label.loc[
        best_models_by_label.groupby("label").idxmax()["f5"]
    ]
    best_models_by_label = best_models_by_label.set_index(["label", "model"])
    best_model_by_label = best_model_by_label.set_index(["label", "model"])
    with pd.option_context(
        "display.max_rows",
        None,
        "display.float_format",
        lambda x: "%.3f" % x,
    ):
        print(df.set_index(["dataset", "model", "label"]), end="\n\n")
        print(best_models_by_label, end="\n\n")
        print(dataset_scores, end="\n\n")
        print(best_models, end="\n\n")
        print(best_model_by_label, end="\n\n")
        print(performances.sort_values("samples/s", ascending=False))

    pass


if __name__ == "__main__":
    evaluate_all()
