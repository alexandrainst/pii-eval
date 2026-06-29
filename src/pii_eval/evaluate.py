from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.progress import MofNCompleteColumn, Progress

from pii_eval.constants import (
    ALL_ISO_LABELS,
    DATASET_LABEL_MAP,
    PII_EVAL_CACHE_PATH,
    get_model_label_map,
)
from pii_eval.data import PiiDataset

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
    ground_truth: PiiDataset,
    predictions: PiiDataset,
    progress: Progress | None = None,
    maximum_entries: int | None = None,
) -> tuple[ConfusionResult, dict[str, ConfusionResult], dict[str, ConfusionResult]]:
    if maximum_entries is None:
        maximum_entries = len(ground_truth)
    dataset_name = ground_truth.name
    model_name = predictions.meta.get("model_name", "unknown_model")
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

    labels = set(model_label_map.values()).intersection(dataset_label_map.values())
    if labels:
        labels.add(ALL_LABEL)

    entries_task = progress.add_task(
        "", total=(len(common_iso_labels) + 1) * len(ground_truth)
    )
    all_confusions = {}

    for label in labels:
        for true_entry in ground_truth[:maximum_entries]:
            progress.update(
                entries_task,
                advance=1,
                description=f"Entry: {str(true_entry.uid)[:10]:<10}",
                transient=True,
            )
            pred_entry = predictions.get_uid(true_entry.uid)
            if pred_entry is None:
                continue
            confusion = ConfusionResult.zero()
            assert len(true_entry.bio) == len(pred_entry.bio)
            for true_span, pred_span in zip(true_entry.bio, pred_entry.bio):
                assert (true_span.start, true_span.end) == (
                    pred_span.start,
                    pred_span.end,
                )
                true_iso_label = _get_iso_label(true_span.label, dataset_label_map)
                pred_iso_label = _get_iso_label(pred_span.label, model_label_map)
                if (
                    true_iso_label
                    and true_iso_label in model_label_map.values()
                    and label in [true_iso_label, ALL_LABEL]
                ):
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
    progress.remove_task(entries_task)

    return all_confusions


def evaluate_all(
    base_directory: Path | str = f"{PII_EVAL_CACHE_PATH}/predictions/",
    write_cache: bool = False,
    maximum_entries: int | None = None,
):
    base_directory = Path(base_directory)
    results = []
    performances = []
    prediction_paths = list(base_directory.glob("*/*.pickle"))
    print(
        f"Calculating evaluation metrics ({[p.stem for p in prediction_paths]})",
        end="\n\n",
    )
    progress = Progress(*Progress.get_default_columns(), MofNCompleteColumn())
    progress.start()
    predictions_task = progress.add_task(
        "", total=len(prediction_paths), transient=True
    )
    for predictions_path in prediction_paths:
        dataset_name = predictions_path.stem
        model_name = predictions_path.parent.name
        progress.update(
            predictions_task,
            advance=1,
            description=f"Model: {model_name[:10]:<10}\nDataset: {dataset_name[:10]:<10}",
        )
        ground_truth_path = Path(
            f"{PII_EVAL_CACHE_PATH}/datasets/{dataset_name}.pickle"
        )
        predictions = PiiDataset.load_pickle(predictions_path)
        if not ground_truth_path.exists():
            continue
        ground_truth = PiiDataset.load_pickle(ground_truth_path)
        confusions = evaluate(
            ground_truth=ground_truth,
            predictions=predictions,
            progress=progress,
            maximum_entries=maximum_entries,
        )
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
    progress.remove_task(predictions_task)
    progress.stop()

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
        out_dir = Path(f"{PII_EVAL_CACHE_PATH}/results/{timestamp}")
        out_dir.mkdir(exist_ok=True, parents=True)
        label_df.to_pickle(out_dir / f"label_df.pickle")
        single_df.to_pickle(out_dir / f"single_df.pickle")
        performance_df.to_pickle(out_dir / "performance_df.pickle")

    # NB: grouping label_df should always include the "label" level
    # (otherwise too many true negatives are counted)

    models_datasets_and_labels = add_metric_columns(
        label_df.groupby(level=["model", "dataset", "label"]).sum()
    ).sort_values(["model", "dataset", "f5"], ascending=[True, True, False])

    models_and_labels = add_metric_columns(
        label_df.groupby(level=["model", "label"]).sum()
    ).sort_values(["model", "f5"], ascending=[True, False])

    models_and_datasets_single = add_metric_columns(
        single_df.groupby(level=["model", "dataset"]).sum()
    ).sort_values(["model", "f5"], ascending=[True, False])

    models_single = add_metric_columns(
        single_df.groupby(level="model").sum()
    ).sort_values("f5", ascending=False)

    models_mean_performance = (
        performance_df.groupby(level="model").mean().sort_values("time_per_entry")
    )

    print(models_and_labels, end="\n\n")
    print(models_and_datasets_single, end="\n\n")
    print(models_single, end="\n\n")
    print(models_mean_performance, end="\n\n")

    pass


if __name__ == "__main__":
    evaluate_all(write_cache=False)
