#!/usr/bin/env python3
"""Upload DIMMI ground-truth data to a private Hugging Face dataset.

This script reads data/ground_truth/dimmi.json, converts it to a
Hugging Face-friendly tabular format, and uploads it to `jharder/dimmi`.

Authentication:
- Uses `python-dotenv` to load environment variables.
- Requires `HF_TOKEN` to be set in `.env` or the shell environment.

Storage format:
- Uses `datasets.DatasetDict.push_to_hub`, which writes parquet shards
  for efficient, compressed transfer and download.
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


def default_data_file() -> Path:
    # Script is stored in src/scripts/, so parents[2] is repository root.
    return Path(__file__).resolve().parents[2] / "data" / "ground_truth" / "dimmi.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload DIMMI data to Hugging Face Hub as a private dataset."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=default_data_file(),
        help="Path to dimmi.json in pii_eval format.",
    )
    parser.add_argument(
        "--repo-id",
        default="jharder/dimmi",
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
        "--public",
        action="store_true",
        help="Upload as public dataset (default is private).",
    )
    return parser.parse_args()


def normalize_bio(raw_bio: Any) -> list[dict[str, Any]]:
    """Normalize raw `bio` spans to a stable schema for HF features."""
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


def build_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append(
            {
                "uid": str(entry.get("uid", "")),
                "text": str(entry.get("text", "")),
                "bio": normalize_bio(entry.get("bio", [])),
                # Store potentially heterogeneous metadata as JSON.
                # "meta": json.dumps(entry.get("meta", {}), ensure_ascii=False),
            }
        )
    return rows


def build_dataset(rows: list[dict[str, Any]], split: str) -> DatasetDict:
    features = Features(
        {
            "uid": Value("string"),
            "text": Value("string"),
            # A list of span structs (list[dict]) instead of a dict-of-lists.
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


def main() -> None:
    args = parse_args()
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise SystemExit(
            "Missing HF_TOKEN. Set it in your shell or in a .env file before running."
        )

    if not args.data_file.exists():
        raise SystemExit(f"Data file not found: {args.data_file}")

    payload = json.loads(args.data_file.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("Invalid dimmi.json format: expected 'entries' to be a list.")

    rows = build_rows(entries)
    ds_dict = build_dataset(rows=rows, split=args.split)

    is_private = not args.public
    print(
        f"Uploading {len(rows)} rows to {args.repo_id} "
        f"(split={args.split}, private={is_private}) ..."
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
