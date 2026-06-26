#!/usr/bin/env python3
"""Upload merged synthetic PII datasets to Hugging Face Hub.

This script reads:
- data/ground_truth/gen_prompt.json
- data/ground_truth/gen_presidio.json

It merges both into a single Hugging Face dataset and uploads it to
`jharder/pii-synth-alpha` as a public dataset.

Authentication:
- Uses `python-dotenv` to load environment variables.
- Requires `HF_TOKEN` to be set in `.env` or shell environment.

Storage format:
- Uses `datasets.DatasetDict.push_to_hub`, which writes parquet shards for
  efficient compressed upload/download.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi


def default_ground_truth_file(filename: str) -> Path:
    # Script is stored in src/scripts/, so parents[2] is repository root.
    return Path(__file__).resolve().parents[2] / "data" / "ground_truth" / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge gen_prompt.json and gen_presidio.json and upload as "
            "a public Hugging Face dataset."
        )
    )
    parser.add_argument(
        "--gen-prompt-file",
        type=Path,
        default=default_ground_truth_file("gen_prompt.json"),
        help="Path to gen_prompt.json in pii_eval format.",
    )
    parser.add_argument(
        "--gen-presidio-file",
        type=Path,
        default=default_ground_truth_file("gen_presidio.json"),
        help="Path to gen_presidio.json in pii_eval format.",
    )
    parser.add_argument(
        "--repo-id",
        default="jharder/pii-synth-alpha",
        help="Target Hugging Face dataset repo (owner/name).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name used on upload.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="200MB",
        help="Maximum parquet shard size for push_to_hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload as private (default is public).",
    )
    return parser.parse_args()


def normalize_bio(raw_bio: Any) -> list[dict[str, Any]]:
    """Normalize raw bio spans to a stable schema for HF features."""
    if not isinstance(raw_bio, list):
        return []

    normalized = []
    for span in raw_bio:
        if not isinstance(span, dict):
            continue
        normalized.append(
            {
                "start": int(span.get("start", 0)),
                "end": int(span.get("end", 0)),
                "text": str(span.get("text", "")),
                "label": str(span.get("label", "")),
            }
        )
    return normalized


def validate_entries(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(f"Invalid dataset format in {path}: expected 'entries' list.")
    return entries


def build_rows(entries: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        original_uid = str(entry.get("uid", ""))
        # Prefix with source because uid values overlap between source files.
        merged_uid = f"{source}:{original_uid}"
        rows.append(
            {
                "uid": merged_uid,
                "text": str(entry.get("text", "")),
                "bio": normalize_bio(entry.get("bio", [])),
            }
        )
    return rows


def build_dataset(rows: list[dict[str, Any]], split: str) -> DatasetDict:
    features = Features(
        {
            "uid": Value("string"),
            "text": Value("string"),
            "bio": [
                {
                    "start": Value("int64"),
                    "end": Value("int64"),
                    "text": Value("string"),
                    "label": Value("string"),
                }
            ],
        }
    )
    dataset = Dataset.from_list(rows, features=features)
    return DatasetDict({split: dataset})


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise SystemExit(
            "Missing HF_TOKEN. Set it in your shell or in a .env file before running."
        )

    prompt_payload = load_json(args.gen_prompt_file)
    presidio_payload = load_json(args.gen_presidio_file)

    prompt_entries = validate_entries(prompt_payload, args.gen_prompt_file)
    presidio_entries = validate_entries(presidio_payload, args.gen_presidio_file)

    prompt_rows = build_rows(prompt_entries, source="gen_prompt")
    presidio_rows = build_rows(presidio_entries, source="gen_presidio")
    merged_rows = prompt_rows + presidio_rows

    ds_dict = build_dataset(rows=merged_rows, split=args.split)

    is_private = args.private
    print(
        f"Uploading {len(merged_rows)} rows "
        f"(gen_prompt={len(prompt_rows)}, gen_presidio={len(presidio_rows)}) "
        f"to {args.repo_id} (split={args.split}, private={is_private}) ..."
    )

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=args.repo_id,
        private=is_private,
        repo_type="dataset",
        exist_ok=True,
    )

    ds_dict.push_to_hub(
        repo_id=args.repo_id,
        token=hf_token,
        private=is_private,
        max_shard_size=args.max_shard_size,
    )
    print("Upload complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
