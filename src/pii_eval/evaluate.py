from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.progress import track

from pii_eval.constants import (
    ALL_ISO_LABELS,
    DATASET_LABEL_MAP,
    get_model_label_map,
)
from pii_eval.data import PiiDataset
from pii_eval.span import align_bios

CONFUSION_COLUMNS = ["tp", "fp", "fn", "tn"]
METRIC_COLUMNS = ["precision", "f1", "f5", "recall"]
ALL_LABEL = "all"


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
        f1 = (2.0 * c.tp) / f1_denominator if f1_denominator else 0.0

        beta_squared = 25.0
        f5_denominator = ((1.0 + beta_squared) * c.tp) + (beta_squared * c.fn) + c.fp
        f5 = ((1.0 + beta_squared) * c.tp) / f5_denominator if f5_denominator else 0.0

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


def _get_iso_label(tag: str, label_map: dict[str, str]) -> str | None:
    if not tag or tag == "O" or "-" not in tag:
        return None
    _, raw_label = tag.split("-", 1)
    return label_map.get(raw_label)


def evaluate(
    ground_truth: PiiDataset, predictions: PiiDataset
) -> tuple[ConfusionResult, dict[str, ConfusionResult], dict[str, ConfusionResult]]:
    dataset_name = ground_truth.meta.get("dataset_name") or "some dataset"
    model_name = predictions.meta.get("model_name") or "some model"
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

    all_confusions = {}
    for label in common_iso_labels.union([ALL_LABEL]):
        for true_entry in ground_truth:
            pred_entry = predictions.get_uid(true_entry.uid)
            if pred_entry is None:
                continue
            confusion = ConfusionResult.zero()

            for true_span, pred_span in align_bios(true_entry.bio, pred_entry.bio):
                true_iso_label = _get_iso_label(true_span.label, dataset_label_map)
                pred_iso_label = _get_iso_label(pred_span.label, model_label_map)
                if true_iso_label and label == true_iso_label or label:
                    if pred_iso_label and label in [pred_iso_label, ALL_LABEL]:
                        confusion.tp += 1
                    else:
                        confusion.fn += 1
                else:
                    if pred_iso_label and label in [pred_iso_label, ALL_LABEL]:
                        confusion.fp += 1
                    else:
                        confusion.tn += 1
            all_confusions[label, true_entry.uid] = confusion

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
        predictions = PiiDataset.load_json(predictions_path)
        if not ground_truth_path.exists():
            continue
        ground_truth = PiiDataset.load_json(ground_truth_path)
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
        time_per_entry = predictions.meta.get("computation_time_per_entry")
        if time_per_entry:
            performances.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "time_per_entry": time_per_entry,
                }
            )

    performance_df = (
        pd.DataFrame(performances).set_index(["model", "dataset"]).sort_index()
    )
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

    best_models_single = add_metric_columns(
        single_df.groupby(level="model").sum()
    ).sort_values("f5", ascending=False)
    models_and_datasets_single = add_metric_columns(
        single_df.groupby(level=["model", "dataset"]).sum()
    ).sort_values("f5", ascending=False)

    pass


if __name__ == "__main__":
    evaluate_all(base_directory="data/predictions", write_cache=True)
