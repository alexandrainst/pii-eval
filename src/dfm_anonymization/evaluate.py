import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import pickle
from statistics import mean
from typing import Dict, List

import pandas as pd
from rich.progress import track

from dfm_anonymization.constants import (
    ALL_ISO_LABELS,
    DATASET_LABEL_MAP,
    MODEL_LABEL_MAP,
    get_model_label_map,
)
from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.util.util import to_dict_recursive


CONFUSION_COLUMNS = ["tp", "fp", "fn", "tn"]
METRIC_COLUMNS = ["precision", "f1", "f5", "recall"]
ALL_LABEL = "all"


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
class EvaluationResult(ConfusionResult):
    dataset: str
    model: str
    label: str


def add_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in CONFUSION_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Dataframe is missing confusion result columns: {missing_columns!r}"
        )

    result = df.copy()
    tp = result["tp"]
    fp = result["fp"]
    fn = result["fn"]

    result["precision"] = (tp / (tp + fp)).fillna(0.0)
    result["f1"] = ((2.0 * tp) / ((2.0 * tp) + fp + fn)).fillna(0.0)

    beta_squared = 25.0
    result["f5"] = (
        ((1.0 + beta_squared) * tp)
        / (((1.0 + beta_squared) * tp) + (beta_squared * fn) + fp)
    ).fillna(0.0)
    result["recall"] = (tp / (tp + fn)).fillna(0.0)

    return result


def _index_entries_by_id(dataset: PiiDataset, dataset_role: str) -> list[PiiDataEntry]:
    indexed_entries = {}

    for index, entry in enumerate(dataset):
        entry_id = entry.id if entry.id is not None else index
        if entry_id in indexed_entries:
            raise ValueError(
                f"Duplicate entry id {entry_id!r} found in {dataset_role}."
            )
        indexed_entries[entry_id] = entry

    return indexed_entries


def _get_aligned_entries(
    ground_truth: PiiDataset, predictions: PiiDataset
) -> list[tuple[PiiDataEntry, PiiDataEntry]]:
    ground_truth_by_id = _index_entries_by_id(ground_truth, "ground truth")
    predictions_by_id = _index_entries_by_id(predictions, "predictions")

    unknown_prediction_ids = set(predictions_by_id).difference(ground_truth_by_id)
    if unknown_prediction_ids:
        sample_ids = sorted(unknown_prediction_ids)[:5]
        raise ValueError(
            f"Predictions contain entry ids not present in ground truth: {sample_ids!r}"
        )

    common_ids = [
        entry_id for entry_id in ground_truth_by_id if entry_id in predictions_by_id
    ]
    if not common_ids:
        raise ValueError("Ground truth and predictions do not share any entry ids.")

    return [
        (ground_truth_by_id[entry_id], predictions_by_id[entry_id])
        for entry_id in common_ids
    ]


def _get_iso_label(tag: str, label_map: Dict[str, str | None]) -> str | None:
    if not tag or tag == "O" or "-" not in tag:
        return None

    _, raw_label = tag.split("-", 1)
    return label_map.get(raw_label)


def evaluate(
    ground_truth: PiiDataset, predictions: PiiDataset
) -> tuple[ConfusionResult, dict[str, ConfusionResult], dict[str, ConfusionResult]]:
    dataset_name = ground_truth.metadata.get("dataset_name") or "some dataset"
    model_name = predictions.metadata.get("model_name") or "some model"
    dataset_label_map = (
        DATASET_LABEL_MAP[dataset_name]
        if dataset_name in DATASET_LABEL_MAP
        else {k: k for k in ALL_ISO_LABELS}
    )
    model_label_map = get_model_label_map(
        model_name, default={k: k for k in ALL_ISO_LABELS}
    )
    common_iso_labels = set(dataset_label_map.values()).intersection(
        model_label_map.values()
    )
    aligned_entries = _get_aligned_entries(ground_truth, predictions)

    all_confusions = {}
    for label in common_iso_labels.union([ALL_LABEL]):
        for true_entry, pred_entry in aligned_entries:
            confusion = ConfusionResult.zero()
            if true_entry.tokens != pred_entry.tokens:
                raise ValueError(
                    f"Token mismatch for entry id {true_entry.id!r}; evaluation requires identical tokenization."
                )
            if len(true_entry.bio_tags) != len(pred_entry.bio_tags):
                raise ValueError(
                    f"Tag length mismatch for entry id {true_entry.id!r}: "
                    f"{len(true_entry.bio_tags)} != {len(pred_entry.bio_tags)}"
                )

            for true_tag, pred_tag in zip(true_entry.bio_tags, pred_entry.bio_tags):
                true_iso_label = _get_iso_label(true_tag, dataset_label_map)
                pred_iso_label = _get_iso_label(pred_tag, model_label_map)
                if true_iso_label and label in [true_iso_label, ALL_LABEL]:
                    if pred_iso_label and label in [pred_iso_label, ALL_LABEL]:
                        confusion.tp += 1
                    else:
                        confusion.fn += 1
                else:
                    if pred_iso_label and label in [pred_iso_label, ALL_LABEL]:
                        confusion.fp += 1
                    else:
                        confusion.tn += 1
            all_confusions[label, true_entry.id] = confusion

    return all_confusions


def evaluate_all(
    base_directory: Path | str = "data/predictions", write_cache: bool = False
):
    if type(base_directory) == str:
        base_directory = Path(base_directory)
    results = []
    performances = []
    prediction_paths = list(base_directory.glob("*/*.json"))
    for predictions_path in track(prediction_paths):
        dataset_name = predictions_path.stem
        model_name = predictions_path.parent.name
        ground_truth_path = Path(f"data/ground_truth/{dataset_name}.json")
        predictions = PiiDataset.from_file(predictions_path)
        if not ground_truth_path.exists():
            continue
        ground_truth = PiiDataset.from_file(ground_truth_path)
        confusions = evaluate(ground_truth=ground_truth, predictions=predictions)
        results += [
            {
                "model": model_name,
                "dataset": dataset_name,
                "entry": entry_id,
                "label": label,
                **asdict(confusion),
            }
            for (label, entry_id), confusion in confusions.items()
        ]
        time_per_entry = predictions.metadata.get("computation_time_per_entry")
        if time_per_entry:
            performances.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "time_per_entry": time_per_entry,
                }
            )
    performance_df = pd.DataFrame(performances)
    df = (
        pd.DataFrame(results)
        .set_index(["model", "dataset", "entry", "label"])
        .sort_index()
    )
    label_df = df[df.index.get_level_values("label") != ALL_LABEL]
    single_df = df[df.index.get_level_values("label") == ALL_LABEL].droplevel("label")

    if write_cache:
        timestamp = datetime.now().isoformat()
        out_dir = Path(f"data/results/{timestamp}")
        out_dir.mkdir(exist_ok=True, parents=True)
        label_df.to_csv(out_dir / "label.csv")
        single_df.to_csv(out_dir / "single.csv")
        performance_df.to_csv(out_dir / "performance.csv")

    # best_models = add_metric_columns(
    #     label_df.groupby("model")[CONFUSION_COLUMNS].sum()
    # ).sort_values("f5", ascending=False)
    # dataset_scores = add_metric_columns(label_df.groupby("dataset")[CONFUSION_COLUMNS].sum())
    # models_and_labels = add_metric_columns(
    #     label_df.groupby(["model", "label"])[CONFUSION_COLUMNS].sum().reset_index()
    # )
    # best_model_by_label = models_and_labels.loc[
    #     models_and_labels.groupby("label").idxmax()["f5"]
    # ].set_index(["label", "model"])
    # labels_and_models = models_and_labels.set_index(["label", "model"]).sort_values(
    #     ["label", "f5"], ascending=[True, False]
    # )
    # models_and_labels = models_and_labels.set_index(["model", "label"]).sort_values(
    #     ["model", "f5"], ascending=[True, False]
    # )
    # label_df = label_df.set_index(["model", "dataset", "label"]).sort_values(
    #     ["model", "dataset", "f5"], ascending=[True, True, False]
    # )

    single_metrics = add_metric_columns(single_df)

    best_models_single = add_metric_columns(
        single_df.groupby(level="model").sum()
    ).sort_values("f5", ascending=False)
    best_models_by_dataset_single = add_metric_columns(
        single_df.groupby(level=["model", "dataset"]).sum()
    ).sort_values(["dataset", "f5"], ascending=[True, False])

    best_feasible_models_by_dataset = best_models_by_dataset_single.loc[
        best_models_by_dataset_single["recall"] >= 0.95
    ].sort_values("precision", ascending=False)

    best_feasible_models = best_models_single.loc[
        best_models_single["recall"] >= 0.95
    ].sort_values("precision", ascending=False)

    feasible_models = set(best_feasible_models.index)

    worst_by_model_and_dataset_single = single_metrics.loc[
        single_metrics.groupby(level=["model", "dataset"]).idxmin()["recall"]
    ]

    worst_by_model_single = single_metrics.loc[
        single_metrics.groupby(level=["model"]).idxmin()["recall"]
    ]

    worst_by_feasible_model_and_dataset_single = worst_by_model_and_dataset_single.loc[
        worst_by_model_and_dataset_single.index.get_level_values("model").isin(
            feasible_models
        )
    ]

    worst_by_feasible_model_single = worst_by_model_single.loc[
        worst_by_model_single.index.get_level_values("model").isin(feasible_models)
    ]

    single_df_feasible = add_metric_columns(
        single_df.loc[single_df.index.get_level_values("model").isin(feasible_models)]
    )

    for_pres_dir = Path("docs/for_pres")
    best_feasible_models[["precision", "recall"]].to_markdown(
        for_pres_dir / "best_feasible_models.md"
    )
    worst_by_feasible_model_and_dataset_single[["precision", "recall"]].to_markdown(
        for_pres_dir / "worst_by_feasible_model_and_dataset.md"
    )

    pass

    # models_and_datasets = add_metric_columns(single_df.
    #     .groupby(["model", "dataset"])[CONFUSION_COLUMNS]
    #     .sum()
    # ).sort_values(["dataset", "precision"], ascending=[True, False])
    # # models_and_datasets = models_and_datasets[models_and_datasets["recall"] >= 0.95]

    # models_only = add_metric_columns(
    #     pd.DataFrame(to_dict_recursive(results_one_label))
    #     .groupby(["model"])[CONFUSION_COLUMNS + ["worst"]]
    #     .sum()
    # ).sort_values(["precision"], ascending=False)
    # models_only = models_only[models_only["recall"] >= 0.95]

    # with pd.option_context(
    #     "display.max_rows",
    #     None,
    #     "display.max_columns",
    #     None,
    #     "display.float_format",
    #     lambda x: "%.3f" % x,
    # ):
    #     # print(df, end="\n\n")
    #     # print(dataset_scores, end="\n\n")
    #     # print(models_and_labels, end="\n\n")
    #     # print(labels_and_models, end="\n\n")
    #     # if len(best_models) > 1:
    #     #     print(best_model_by_label, end="\n\n")
    #     # print(best_models, end="\n\n")
    #     print(performance_df.sort_values("time_per_entry", ascending=False), end="\n\n")

    #     print("Treating all labels as one:")
    #     print(models_and_datasets, end="\n\n")
    #     print(models_only)


if __name__ == "__main__":
    evaluate_all(base_directory="data/predictions", write_cache=True)
