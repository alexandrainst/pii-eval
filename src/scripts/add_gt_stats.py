#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def iter_dataset_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def normalize_label(label: str) -> str:
    if "-" in label:
        prefix, remainder = label.split("-", 1)
        if prefix in {"B", "I"} and remainder:
            return remainder
    return label


def iter_entry_labels(entry: dict) -> Iterable[str]:
    bio = entry.get("bio")
    if isinstance(bio, list):
        for token in bio:
            if not isinstance(token, dict):
                continue
            label = token.get("label")
            if isinstance(label, str) and label:
                yield normalize_label(label)

    # Support legacy datasets that store labels as flat BIO tags.
    bio_tags = entry.get("bio_tags")
    if isinstance(bio_tags, list):
        for tag in bio_tags:
            if isinstance(tag, str) and tag:
                yield normalize_label(tag)


def count_labels(entries: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        counter.update(iter_entry_labels(entry))
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def add_stats_to_dataset(data: dict) -> dict:
    entries = data.get("entries")
    if not isinstance(entries, list):
        entries = []

    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    updated_meta = dict(meta)
    updated_meta["labels"] = count_labels(entries)

    # Preserve deterministic top-level ordering with meta immediately after name.
    updated: dict = {}
    if "name" in data:
        updated["name"] = data["name"]
    updated["meta"] = updated_meta
    for key, value in data.items():
        if key in {"name", "meta"}:
            continue
        updated[key] = value
    return updated


def process_dataset_file(dataset_path: Path, output_dir: Path) -> tuple[Path, int]:
    raw = dataset_path.read_text(encoding="utf8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON value must be an object in {dataset_path}")

    updated = add_stats_to_dataset(data)
    output_path = output_dir / dataset_path.name
    output_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8",
    )

    labels = updated.get("meta", {}).get("labels", {})
    label_count = len(labels) if isinstance(labels, dict) else 0
    return output_path, label_count


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Add top-level label statistics to each dataset and write results to a new directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "ground_truth",
        help="Directory containing input dataset JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "ground_truth_stats",
        help="Directory where output JSON files will be written.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    dataset_paths = iter_dataset_paths(input_dir)
    if not dataset_paths:
        raise SystemExit(f"No JSON files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(dataset_paths)} dataset file(s) from {input_dir}...")
    for dataset_path in dataset_paths:
        output_path, label_count = process_dataset_file(dataset_path, output_dir)
        print(
            f"Wrote {output_path.name} with {label_count} unique labels to {output_path.parent}"
        )


if __name__ == "__main__":
    main()
