from pathlib import Path

import pandas as pd
from dfm_anonymization.inference.dataset import PiiDataset
from dfm_anonymization.util.util import to_dict_recursive


class EvaluationResult:
    precision: float
    f1: float
    f5: float
    recall: float


def evaluate(
    ground_truth: PiiDataset, predictions: PiiDataset
) -> EvaluationResult:
    # TODO get evaluation result from comparing BIO tags
    raise NotImplemented
    return EvaluationResult


def evaluate_all():
    results = []
    for predictions_path in Path("data/infer_ds").rglob("*.json"):
        dataset_name = predictions_path.stem
        ground_truth_path = f"data/ground_truth/{dataset_name}"
        predictions = PiiDataset.from_file(predictions_path)
        ground_truth = PiiDataset.from_file(ground_truth_path)
        results.append(
            evaluate(ground_truth=ground_truth, predictions=predictions))
    df = pd.Dataframe(to_dict_recursive(results))
    print(df)
