#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pii_eval.constants import (  # noqa: E402
    ALL_ISO_LABELS,
    PII_EVAL_CACHE_PATH,
    get_dataset_label_map,
    get_model_label_map,
)
from pii_eval.data import PiiDataset  # noqa: E402


@dataclass(frozen=True)
class SampleCase:
    uid: str
    token_index: int
    start: int
    end: int
    token_text: str
    true_tag: str
    pred_tag: str
    true_iso_label: str | None
    pred_iso_label: str | None
    snippet: str


class ReservoirSampler:
    def __init__(self, size: int, rng: random.Random) -> None:
        self.size = max(size, 0)
        self.rng = rng
        self.total_seen = 0
        self.samples: list[SampleCase] = []

    def add(self, case: SampleCase) -> None:
        self.total_seen += 1
        if self.size <= 0:
            return

        if len(self.samples) < self.size:
            self.samples.append(case)
            return

        replacement_index = self.rng.randrange(self.total_seen)
        if replacement_index < self.size:
            self.samples[replacement_index] = case


@dataclass
class PairReport:
    model_name: str
    model_path: str
    dataset_name: str
    prediction_path: Path
    ground_truth_path: Path
    common_labels: set[str]
    entries_ground_truth: int
    entries_predictions: int
    entries_compared: int
    missing_prediction_entries: int
    misaligned_entries: int
    misaligned_tokens: int
    tokens_compared: int
    tp_sampler: ReservoirSampler
    fp_sampler: ReservoirSampler
    fn_sampler: ReservoirSampler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample false positives, true positives and false negatives "
            "from cached pii_eval predictions."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random samples per category (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(PII_EVAL_CACHE_PATH),
        help="Cache root containing datasets/ and predictions/.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Override predictions directory.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help="Override datasets directory.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit for number of prediction files to process.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search prediction files recursively under predictions-dir.",
    )
    return parser.parse_args()


def _resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _get_iso_label(tag: str, label_map: dict[str, str]) -> str | None:
    if not tag or tag == "O" or "-" not in tag:
        return None
    _, raw = tag.split("-", 1)
    return label_map.get(raw)


def _format_snippet(
    text: str,
    start: int,
    end: int,
    *,
    side_chars: int = 72,
    max_chars: int = 220,
) -> str:
    if start < 0 or end < 0 or start >= end or end > len(text):
        raw = text.replace("\n", " ").strip()
        return raw[:max_chars] + ("..." if len(raw) > max_chars else "")

    left = max(0, start - side_chars)
    right = min(len(text), end + side_chars)

    snippet = text[left:start] + "[[" + text[start:end] + "]]" + text[end:right]
    snippet = snippet.replace("\n", " ").strip()

    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3] + "..."

    return snippet


def _make_case(
    *,
    uid: int | str,
    token_index: int,
    text: str,
    start: int,
    end: int,
    true_tag: str,
    pred_tag: str,
    true_iso_label: str | None,
    pred_iso_label: str | None,
) -> SampleCase:
    safe_start = max(0, min(start, len(text)))
    safe_end = max(safe_start, min(end, len(text)))

    return SampleCase(
        uid=str(uid),
        token_index=token_index,
        start=start,
        end=end,
        token_text=text[safe_start:safe_end],
        true_tag=true_tag,
        pred_tag=pred_tag,
        true_iso_label=true_iso_label,
        pred_iso_label=pred_iso_label,
        snippet=_format_snippet(text, safe_start, safe_end),
    )


def _collect_pair_report(
    *,
    model_name: str,
    model_path: str,
    dataset_name: str,
    prediction_path: Path,
    ground_truth_path: Path,
    predictions: PiiDataset,
    ground_truth: PiiDataset,
    dataset_label_map: dict[str, str],
    model_label_map: dict[str, str],
    sample_size: int,
    rng: random.Random,
) -> PairReport:
    common_labels = set(dataset_label_map.values()).intersection(
        model_label_map.values()
    )

    tp_sampler = ReservoirSampler(size=sample_size, rng=rng)
    fp_sampler = ReservoirSampler(size=sample_size, rng=rng)
    fn_sampler = ReservoirSampler(size=sample_size, rng=rng)

    entries_compared = 0
    missing_prediction_entries = 0
    misaligned_entries = 0
    misaligned_tokens = 0
    tokens_compared = 0

    pred_by_uid = {entry.uid: entry for entry in predictions.entries}

    for true_entry in ground_truth.entries:
        pred_entry = pred_by_uid.get(true_entry.uid)
        if pred_entry is None:
            missing_prediction_entries += 1
            continue

        if len(true_entry.bio) != len(pred_entry.bio):
            misaligned_entries += 1
            continue

        entries_compared += 1

        for token_index, (true_token, pred_token) in enumerate(
            zip(true_entry.bio, pred_entry.bio)
        ):
            if (true_token.start, true_token.end) != (pred_token.start, pred_token.end):
                misaligned_tokens += 1
                continue

            tokens_compared += 1

            true_iso_label = _get_iso_label(true_token.label, dataset_label_map)
            pred_iso_label = _get_iso_label(pred_token.label, model_label_map)

            true_eval_label = (
                true_iso_label if true_iso_label in common_labels else None
            )
            pred_eval_label = (
                pred_iso_label if pred_iso_label in common_labels else None
            )

            needs_tp = (
                true_eval_label is not None and pred_eval_label == true_eval_label
            )
            needs_fp = (
                pred_eval_label is not None and pred_eval_label != true_eval_label
            )
            needs_fn = (
                true_eval_label is not None and pred_eval_label != true_eval_label
            )

            if not (needs_tp or needs_fp or needs_fn):
                continue

            case = _make_case(
                uid=true_entry.uid,
                token_index=token_index,
                text=true_entry.text,
                start=true_token.start,
                end=true_token.end,
                true_tag=true_token.label,
                pred_tag=pred_token.label,
                true_iso_label=true_eval_label,
                pred_iso_label=pred_eval_label,
            )

            if needs_tp:
                tp_sampler.add(case)
            if needs_fp:
                fp_sampler.add(case)
            if needs_fn:
                fn_sampler.add(case)

    return PairReport(
        model_name=model_name,
        model_path=model_path,
        dataset_name=dataset_name,
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
        common_labels=common_labels,
        entries_ground_truth=len(ground_truth.entries),
        entries_predictions=len(predictions.entries),
        entries_compared=entries_compared,
        missing_prediction_entries=missing_prediction_entries,
        misaligned_entries=misaligned_entries,
        misaligned_tokens=misaligned_tokens,
        tokens_compared=tokens_compared,
        tp_sampler=tp_sampler,
        fp_sampler=fp_sampler,
        fn_sampler=fn_sampler,
    )


def _print_category(title: str, sampler: ReservoirSampler) -> None:
    print("-" * 100)
    print(f"{title} (showing {len(sampler.samples)} of {sampler.total_seen})")
    if sampler.total_seen == 0:
        print("  (none)")
        return

    for index, case in enumerate(sampler.samples, start=1):
        print(
            f"{index:>2}. uid={case.uid} token_idx={case.token_index} "
            f"span=[{case.start}:{case.end}] token={case.token_text!r}"
        )
        print(
            f"    true={case.true_tag} -> {case.true_iso_label or 'O'} | "
            f"pred={case.pred_tag} -> {case.pred_iso_label or 'O'}"
        )
        print(f"    context: {case.snippet}")


def _print_pair_report(report: PairReport, index: int, total: int) -> None:
    print("\n" + "=" * 100)
    print(
        f"[{index}/{total}] MODEL={report.model_path} "
        f"(label-map key: {report.model_name}) | DATASET={report.dataset_name}"
    )
    print("=" * 100)
    print(f"Prediction file : {report.prediction_path}")
    print(f"Ground truth    : {report.ground_truth_path}")

    labels_text = ", ".join(sorted(report.common_labels))
    if not labels_text:
        labels_text = "(none)"
    print(f"Common labels   : {labels_text}")

    print(
        "Entries         : "
        f"ground_truth={report.entries_ground_truth}, "
        f"predictions={report.entries_predictions}, "
        f"compared={report.entries_compared}, "
        f"missing_prediction={report.missing_prediction_entries}, "
        f"misaligned_entries={report.misaligned_entries}"
    )
    print(
        "Tokens          : "
        f"compared={report.tokens_compared}, "
        f"misaligned_tokens={report.misaligned_tokens}, "
        f"TP={report.tp_sampler.total_seen}, "
        f"FP={report.fp_sampler.total_seen}, "
        f"FN={report.fn_sampler.total_seen}"
    )

    _print_category("TRUE POSITIVES", report.tp_sampler)
    _print_category("FALSE POSITIVES", report.fp_sampler)
    _print_category("FALSE NEGATIVES", report.fn_sampler)


def main() -> None:
    args = _parse_args()

    if args.samples < 0:
        raise ValueError("--samples must be >= 0")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be > 0")

    cache_root = _resolve_path(args.cache_root, REPO_ROOT)
    predictions_dir = (
        _resolve_path(args.predictions_dir, REPO_ROOT)
        if args.predictions_dir is not None
        else cache_root / "predictions"
    )
    datasets_dir = (
        _resolve_path(args.datasets_dir, REPO_ROOT)
        if args.datasets_dir is not None
        else cache_root / "datasets"
    )

    if not predictions_dir.exists():
        print(f"Predictions directory does not exist: {predictions_dir}")
        return
    if not datasets_dir.exists():
        print(f"Datasets directory does not exist: {datasets_dir}")
        return

    if args.recursive:
        prediction_paths = sorted(
            path for path in predictions_dir.rglob("*.pickle") if path.is_file()
        )
    else:
        prediction_paths = sorted(
            path for path in predictions_dir.glob("*/*.pickle") if path.is_file()
        )
    if args.max_pairs is not None:
        prediction_paths = prediction_paths[: args.max_pairs]

    if not prediction_paths:
        print(f"No prediction pickle files found under: {predictions_dir}")
        return

    print("=" * 100)
    print("Sampling TP / FP / FN from cached predictions")
    print("=" * 100)
    print(f"Cache root      : {cache_root}")
    print(f"Predictions dir : {predictions_dir}")
    print(f"Datasets dir    : {datasets_dir}")
    print(f"Prediction files: {len(prediction_paths)}")
    print(f"Sample size     : {args.samples}")
    print(f"Seed            : {args.seed}")

    identity_map = {label: label for label in ALL_ISO_LABELS}
    rng = random.Random(args.seed)

    processed = 0
    skipped_missing_ground_truth = 0
    failed = 0

    for index, prediction_path in enumerate(prediction_paths, start=1):
        relative_parent = prediction_path.relative_to(predictions_dir).parent
        model_path = (
            relative_parent.as_posix() if str(relative_parent) != "." else "<root>"
        )
        model_name = prediction_path.parent.name
        dataset_name = prediction_path.stem
        ground_truth_path = datasets_dir / f"{dataset_name}.pickle"

        if not ground_truth_path.exists():
            skipped_missing_ground_truth += 1
            print("\n" + "=" * 100)
            print(
                f"[{index}/{len(prediction_paths)}] SKIPPED MODEL={model_path} DATASET={dataset_name}"
            )
            print("=" * 100)
            print(f"Missing ground-truth pickle: {ground_truth_path}")
            continue

        try:
            predictions = PiiDataset.load_pickle(prediction_path)
            ground_truth = PiiDataset.load_pickle(ground_truth_path)

            dataset_label_map = (
                get_dataset_label_map(dataset_name, default=identity_map)
                or identity_map
            )
            model_label_map = (
                get_model_label_map(model_name, default=identity_map) or identity_map
            )

            report = _collect_pair_report(
                model_name=model_name,
                model_path=model_path,
                dataset_name=dataset_name,
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
                predictions=predictions,
                ground_truth=ground_truth,
                dataset_label_map=dataset_label_map,
                model_label_map=model_label_map,
                sample_size=args.samples,
                rng=rng,
            )
            _print_pair_report(report, index=index, total=len(prediction_paths))
            processed += 1
        except Exception as exc:
            failed += 1
            print("\n" + "=" * 100)
            print(
                f"[{index}/{len(prediction_paths)}] FAILED MODEL={model_path} DATASET={dataset_name}"
            )
            print("=" * 100)
            print(f"Prediction file : {prediction_path}")
            print(f"Ground truth    : {ground_truth_path}")
            print(f"Error           : {type(exc).__name__}: {exc}")

    print("\n" + "=" * 100)
    print("Done")
    print("=" * 100)
    print(
        f"Processed={processed}, "
        f"SkippedMissingGroundTruth={skipped_missing_ground_truth}, "
        f"Failed={failed}, "
        f"TotalFiles={len(prediction_paths)}"
    )


if __name__ == "__main__":
    main()
