#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pii_eval.data import PiiDataset
from pii_eval.span import WordTokenizer, bio_to_spans, spans_to_bio


@dataclass
class DatasetSummary:
    path: Path
    total_entries: int
    wrong_tokenizations: int


def default_input_dir() -> Path:
    return Path(__file__).resolve().parents[2] / ".pii_eval_cache" / "datasets"


def iter_dataset_pickles(input_dir: Path, pattern: str = "*.pickle") -> list[Path]:
    return sorted(path for path in input_dir.glob(pattern=pattern) if path.is_file())


def has_correct_tokenization(entry, tokenizer: WordTokenizer) -> bool:
    expected_tokens = tokenizer(entry.text)
    actual_tokens = entry.bio

    if len(expected_tokens) != len(actual_tokens):
        return False

    for expected_token, actual_token in zip(expected_tokens, actual_tokens):
        if expected_token.start != actual_token.start:
            return False
        if expected_token.end != actual_token.end:
            return False
        if expected_token.text != actual_token.text:
            return False

    return True


def summarize_dataset(path: Path, tokenizer: WordTokenizer) -> DatasetSummary:
    dataset = PiiDataset.load_pickle(path)
    wrong_tokenizations = sum(
        1
        for entry in dataset.entries
        if not has_correct_tokenization(entry=entry, tokenizer=tokenizer)
    )

    return DatasetSummary(
        path=path,
        total_entries=len(dataset.entries),
        wrong_tokenizations=wrong_tokenizations,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize tokenization mismatches in cached dataset pickles by "
            "comparing entry.bio token spans to WordTokenizer output."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=default_input_dir(),
        help="Directory containing dataset .pickle files (default: .pii_eval_cache/datasets).",
    )
    return parser.parse_args()


def summerize_wrong_tokenizations(pattern: str = "*.pickle") -> None:
    args = parse_args()
    input_dir = args.input_dir

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    dataset_files = iter_dataset_pickles(input_dir, pattern=pattern)
    if not dataset_files:
        raise SystemExit(f"No .pickle files found in: {input_dir}")

    tokenizer = WordTokenizer()
    summaries = [summarize_dataset(path, tokenizer) for path in dataset_files]

    total_entries = 0
    total_wrong = 0

    for summary in summaries:
        total_entries += summary.total_entries
        total_wrong += summary.wrong_tokenizations
        ratio = (
            (summary.wrong_tokenizations / summary.total_entries * 100.0)
            if summary.total_entries
            else 0.0
        )
        print(
            f"{summary.path.name}: "
            f"{summary.wrong_tokenizations}/{summary.total_entries} "
            f"wrong tokenizations ({ratio:.2f}%)"
        )

    overall_ratio = (total_wrong / total_entries * 100.0) if total_entries else 0.0
    print(
        f"TOTAL: {total_wrong}/{total_entries} wrong tokenizations "
        f"across {len(summaries)} dataset file(s) ({overall_ratio:.2f}%)"
    )


def fix_wrong_tokenizations(pattern: str = "*.pickle"):
    input_dir = default_input_dir()
    dataset_files = iter_dataset_pickles(input_dir=input_dir, pattern=pattern)

    if not dataset_files:
        raise SystemExit(
            f"No .pickle files found in: {input_dir} with pattern: {pattern}"
        )

    tokenizer = WordTokenizer()

    for path in dataset_files:
        dataset = PiiDataset.load_pickle(path)
        fixed_entries = 0

        for entry in dataset.entries:
            if has_correct_tokenization(entry=entry, tokenizer=tokenizer):
                continue

            spans = bio_to_spans(text=entry.text, bio=entry.bio)
            entry.bio = spans_to_bio(
                text=entry.text,
                spans=spans,
                tokenize=tokenizer,
                include_o_labels=True,
            )
            fixed_entries += 1

        dataset.dump_pickle(path)
        print(f"FIXED {path.name}: {fixed_entries}/{len(dataset.entries)} entries")


if __name__ == "__main__":
    summerize_wrong_tokenizations("dane.pickle")
    # fix_wrong_tokenizations("dane.pickle")
