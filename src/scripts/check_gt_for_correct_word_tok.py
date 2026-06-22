#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pii_eval.data import PiiDataset
from pii_eval.span import WordTokenizer, bio_to_spans, spans_to_bio


@dataclass
class FileCheckResult:
	path: Path
	total_entries: int
	corrected_entries: int
	output_path: Path | None


def default_input_dir() -> Path:
	return Path(__file__).resolve().parents[2] / "data" / "ground_truth"


def corrected_output_dir(input_dir: Path) -> Path:
	return input_dir.parent / f"{input_dir.name}_corrected"


def iter_json_files(input_dir: Path) -> list[Path]:
	return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def has_correct_tokenization(entry, tokenizer: WordTokenizer) -> bool:
	expected_tokens = tokenizer(entry.text)
	if len(expected_tokens) != len(entry.bio):
		return False

	for expected_token, actual_token in zip(expected_tokens, entry.bio):
		if expected_token.start != actual_token.start:
			return False
		if expected_token.end != actual_token.end:
			return False
		if expected_token.text != actual_token.text:
			return False

	return True


def corrected_bio(entry, tokenizer: WordTokenizer):
	spans = bio_to_spans(text=entry.text, bio=entry.bio)
	return spans_to_bio(
		text=entry.text,
		spans=spans,
		tokenize=tokenizer,
		include_o_labels=True,
	)


def check_and_correct_file(
	input_path: Path,
	output_dir: Path,
	tokenizer: WordTokenizer,
) -> FileCheckResult:
	dataset = PiiDataset.load_json(input_path)
	corrected_entries = 0

	for entry in dataset.entries:
		if has_correct_tokenization(entry=entry, tokenizer=tokenizer):
			continue
		entry.bio = corrected_bio(entry=entry, tokenizer=tokenizer)
		corrected_entries += 1

	output_path = None
	if corrected_entries > 0:
		output_path = output_dir / input_path.name
		dataset.save_json(output_path)

	return FileCheckResult(
		path=input_path,
		total_entries=len(dataset.entries),
		corrected_entries=corrected_entries,
		output_path=output_path,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Check word-tokenization in dataset JSON files and write corrected "
			"copies for files with mismatches."
		)
	)
	parser.add_argument(
		"input_dir",
		nargs="?",
		type=Path,
		default=default_input_dir(),
		help="Directory containing dataset .json files (default: data/ground_truth).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_dir = args.input_dir

	if not input_dir.exists():
		raise SystemExit(f"Input directory not found: {input_dir}")
	if not input_dir.is_dir():
		raise SystemExit(f"Input path is not a directory: {input_dir}")

	input_files = iter_json_files(input_dir)
	if not input_files:
		raise SystemExit(f"No .json files found in: {input_dir}")

	output_dir = corrected_output_dir(input_dir)
	tokenizer = WordTokenizer()

	results = [
		check_and_correct_file(
			input_path=input_path,
			output_dir=output_dir,
			tokenizer=tokenizer,
		)
		for input_path in input_files
	]

	corrected_files = [result for result in results if result.corrected_entries > 0]
	unchanged_files = [result for result in results if result.corrected_entries == 0]

	for result in corrected_files:
		print(
			f"CORRECTED {result.path.name}: "
			f"{result.corrected_entries}/{result.total_entries} entries -> {result.output_path}"
		)
	for result in unchanged_files:
		print(f"OK {result.path.name}")

	print(
		f"Summary: {len(corrected_files)} corrected file(s), "
		f"{len(unchanged_files)} already correct file(s)."
	)


if __name__ == "__main__":
	main()
