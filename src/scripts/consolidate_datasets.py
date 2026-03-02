#!/usr/bin/env python3
"""
Consolidate multiple generated dataset files into a single file.

This script takes all JSON files matching the pattern data/ground_truth/gen_*.json
and consolidates them into a single file (data/ground_truth/gen_presidio.json),
ensuring that all entry IDs remain unique.
"""

import json
import sys
from pathlib import Path


def consolidate_datasets(
    output_file: str = "data/ground_truth/gen_presidio.json",
) -> None:
    """
    Consolidate all gen_*.json files into a single output file.

    Args:
        output_file: Path to the output consolidated file
    """
    # Get the repository root (script is in src/scripts/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    # Directory containing the data files
    data_dir = repo_root / "data" / "ground_truth"
    output_path = repo_root / output_file

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Find all gen_*.json files (excluding gen_presidio.json if it exists)
    pattern = "gen_*.json"
    files = sorted([f for f in data_dir.glob(pattern) if f.name != "gen_presidio.json"])

    if not files:
        print(f"Error: No files matching pattern '{pattern}' found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to consolidate:")
    for f in files:
        print(f"  - {f.name}")

    # Consolidate all entries
    consolidated_entries = []
    current_id = 0

    for file_path in files:
        print(f"\nProcessing {file_path.name}...")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract entries from this file
            entries = data.get("entries", [])
            print(f"  - Found {len(entries)} entries")

            # Add entries with updated IDs
            for entry in entries:
                new_entry = entry.copy()
                new_entry["id"] = current_id
                consolidated_entries.append(new_entry)
                current_id += 1

        except json.JSONDecodeError as e:
            print(f"  ERROR: Failed to parse JSON: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

    # Create the consolidated data structure
    consolidated_data = {"entries": consolidated_entries}

    # Write to output file
    print(f"\nWriting consolidated data to {output_path.name}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(consolidated_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Successfully created {output_path.name}")
        print(f"  Total entries: {len(consolidated_entries)}")
    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    consolidate_datasets()
