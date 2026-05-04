#!/usr/bin/env python3
"""
Convert DIMMI raw CSV files to the ground truth JSON format.

Reads data/raw/dimmi/cvs.csv and data/raw/dimmi/passport.csv, tokenizes each
document, aligns PII spans to BIO tags, and writes data/ground_truth/dimmi.json.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow importing from the dfm_anonymization package when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dfm_anonymization.util.util import WordTokenizer, spans_to_bio
from dfm_anonymization.util.util import Span


def pii_to_spans(pii_json: str) -> list[Span]:
    """Parse the JSON-encoded PII column into a list of Span objects.

    The CSV stores PII as a JSON array of objects with keys 'category',
    'start', and 'end'. We map 'category' -> 'label' to match the Span model.
    """
    raw = json.loads(pii_json)
    spans = []
    for item in raw:
        spans.append(
            Span(
                start=int(item["start"]),
                end=int(item["end"]),
                label=str(item["category"]),
            )
        )
    return spans


def process_csv(
    csv_path: Path, source_name: str, tokenizer: WordTokenizer
) -> tuple[list[dict], set[str]]:
    """Process a single DIMMI CSV file and return (entries, labels).

    Args:
        csv_path: Path to the CSV file.
        source_name: Short name recorded in each entry's metadata ('cvs' or 'passport').
        tokenizer: Shared WordTokenizer instance.

    Returns:
        A tuple of:
        - list of entry dicts (without final sequential id, id is set by caller)
        - set of label strings seen in this file
    """
    entries = []
    labels: set[str] = set()

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"]
            original_id = row["id"]

            # Tokenize
            tokens, token_spans = tokenizer.word_tokenize(text)

            # Parse PII spans (category -> label)
            spans = pii_to_spans(row["pii"])

            # Collect labels
            for span in spans:
                labels.add(span.label)

            # Align spans to BIO tags
            bio_tags = spans_to_bio(token_spans, spans)

            entries.append(
                {
                    "text": text,
                    "tokens": tokens,
                    "bio_tags": bio_tags,
                    "metadata": {
                        "source": source_name,
                        "original_id": original_id,
                    },
                }
            )

    return entries, labels


def convert_dimmi() -> None:
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    raw_dir = repo_root / "data" / "raw" / "dimmi"

    if not raw_dir.exists():
        print(f"Error: raw data directory not found: {raw_dir}")
        sys.exit(1)

    tokenizer = WordTokenizer()

    sources = [(p, p.stem) for p in raw_dir.glob("*.csv")]

    for csv_path, source_name in sources:

        all_entries: list[dict] = []
        all_labels: set[str] = set()
        output_path = repo_root / "data" / "ground_truth" / f"{source_name}.json"

        if not csv_path.exists():
            print(f"Warning: {csv_path.name} not found, skipping.")
            continue

        print(f"Processing {csv_path.name}...")
        entries, labels = process_csv(csv_path, source_name, tokenizer)
        all_entries.extend(entries)
        all_labels.update(labels)
        print(f"  -> {len(entries)} entries, labels: {sorted(labels)}")

        if not all_entries:
            print("Error: no entries were processed.")
            sys.exit(1)

        # Assign sequential IDs
        for i, entry in enumerate(all_entries):
            entry["id"] = entry["metadata"].pop("original_id")

        # Re-order keys for readability: id first
        ordered_entries = [
            {
                "id": e["id"],
                "text": e["text"],
                "tokens": e["tokens"],
                "bio_tags": e["bio_tags"],
                "metadata": e["metadata"],
            }
            for e in all_entries
        ]

        output = {
            "metadata": {
                "dataset_name": source_name,
                "timestamp": datetime.now().strftime("%Y-%m"),
                "labels": sorted(all_labels),
            },
            "entries": ordered_entries,
        }

        print(f"\nWriting {len(ordered_entries)} entries to {output_path.name}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Done. Output: {output_path}")
        print(f"Total entries : {len(ordered_entries)}")
        print(f"Labels        : {sorted(all_labels)}")


if __name__ == "__main__":
    convert_dimmi()
