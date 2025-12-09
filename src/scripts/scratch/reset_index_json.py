import argparse
import json
from pathlib import Path


def reset_document_ids(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as src:
        data = json.load(src)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array.")

    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError("Each entry must be an object.")
        record["document"] = idx

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as dst:
        json.dump(data, dst, ensure_ascii=True, indent=1)
        dst.write("\n")

    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset 'document' ids in a JSON array file."
    )
    parser.add_argument(
        "--json_path", type=Path, help="Path to the JSON file to update.",
        default="data/infer/dansk_ner/rashmibanthia.json")
    args = parser.parse_args()
    reset_document_ids(args.json_path)


if __name__ == "__main__":
    main()
