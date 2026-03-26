#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_labels(entries: list[dict]) -> list[str]:
	labels = {
		tag.split("-", 1)[1]
		for entry in entries
		for tag in entry.get("bio_tags", [])
		if isinstance(tag, str) and tag not in {"", "O"} and "-" in tag
	}
	return sorted(labels)


def update_dataset_metadata(dataset_path: Path) -> list[str]:
	data = json.loads(dataset_path.read_text(encoding="utf8"))
	metadata = data.get("metadata")
	if not isinstance(metadata, dict):
		metadata = {}

	labels = extract_labels(data.get("entries", []))
	metadata["labels"] = labels
	data["metadata"] = metadata

	dataset_path.write_text(
		json.dumps(data, ensure_ascii=False, indent=2) + "\n",
		encoding="utf8",
	)
	return labels


def iter_ground_truth_datasets(data_dir: Path) -> list[Path]:
	return sorted(path for path in data_dir.glob("*.json") if path.is_file())


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Populate metadata.labels for each ground-truth dataset JSON file."
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path(__file__).resolve().parents[2] / "data" / "ground_truth",
		help="Directory containing ground-truth dataset JSON files.",
	)
	args = parser.parse_args()

	if not args.data_dir.exists():
		raise SystemExit(f"Data directory not found: {args.data_dir}")

	dataset_paths = iter_ground_truth_datasets(args.data_dir)
	if not dataset_paths:
		raise SystemExit(f"No dataset JSON files found in: {args.data_dir}")

	for dataset_path in dataset_paths:
		labels = update_dataset_metadata(dataset_path)
		print(f"Updated {dataset_path.name}: {labels}")


if __name__ == "__main__":
	main()
