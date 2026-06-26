#!/usr/bin/env python3
"""Clean Danish last names file by keeping only surname text per line.

Example input line:
        10. Aagaard-Svendsen (1)

Example output line:
        Aagaard-Svendsen
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# Matches lines like: "123. Name Here (45)" with optional extra whitespace.
LINE_PATTERN = re.compile(r"^\s*\d+\.\s*(.*?)\s*\(\d+\)\s*$")


def extract_last_name(raw_line: str) -> str:
    """Return only the surname part from a source line, or empty string."""
    line = raw_line.strip()
    if not line:
        return ""

    match = LINE_PATTERN.match(line)
    if match:
        return match.group(1).strip()

    # Fallback: strip leading numbering and trailing count independently.
    line = re.sub(r"^\s*\d+\.\s*", "", line)
    line = re.sub(r"\s*\(\d+\)\s*$", "", line)
    return line.strip()


def clean_file(path: Path) -> int:
    """Clean file in place and return the number of written lines."""
    cleaned_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = extract_last_name(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    output = "\n".join(cleaned_lines)
    if cleaned_lines:
        output += "\n"

    path.write_text(output, encoding="utf-8")
    return len(cleaned_lines)


def default_last_names_path() -> Path:
    """Return default path to assets/person_names/last_names_da.txt."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "assets" / "person_names" / "last_names_da.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean last names file by removing indices/counts and empty lines."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=default_last_names_path(),
        help="Path to the source file (cleaned in place).",
    )
    args = parser.parse_args()

    lines_written = clean_file(args.path)
    print(f"Cleaned {lines_written} last names in {args.path}")


if __name__ == "__main__":
    main()
