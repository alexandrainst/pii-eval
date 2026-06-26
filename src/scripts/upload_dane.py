#!/usr/bin/env python3
"""Upload DANE data as a parquet-backed Hugging Face dataset.

This script:
1. Parses CoNLL-U files from data/raw/dane (train/dev/test).
2. Converts token-level NER labels in MISC (name=...) to span-style BIO records.
3. Uploads the dataset to jharder/dane-copy as a public dataset by default.
4. Copies non-data metadata files (README, loader script, license files, etc.)
   from alexandrainst/dane into the target repository.

Authentication:
- Uses python-dotenv to load environment variables.
- Requires HF_TOKEN in .env or shell environment.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download


def default_raw_dir() -> Path:
    # Script is in src/scripts/, so parents[2] is repository root.
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "dane"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DANE CoNLL-U files to HF parquet dataset and upload to Hub."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir(),
        help="Directory containing ddt.train/dev/test.conllu files.",
    )
    parser.add_argument(
        "--source-repo-id",
        default="alexandrainst/dane",
        help="Source dataset repository used for metadata file copying.",
    )
    parser.add_argument(
        "--target-repo-id",
        default="jharder/dane-copy",
        help="Target dataset repository to create/upload.",
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
    parser.add_argument(
        "--skip-metadata-copy",
        action="store_true",
        help="Skip copying non-data metadata files from source repo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build/validate dataset locally; do not upload.",
    )
    return parser.parse_args()


def _parse_name_label(misc_field: str) -> str:
    if not misc_field or misc_field == "_":
        return "O"
    for item in misc_field.split("|"):
        if item.startswith("name="):
            value = item.split("=", 1)[1]
            return value if value else "O"
    return "O"


def _align_tokens_to_text(
    text: str, tokens: list[str], source_path: Path
) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    cursor = 0
    for idx, token in enumerate(tokens):
        start = text.find(token, cursor)
        if start < 0:
            start = text.find(token, max(cursor - 2, 0))
        if start < 0:
            raise ValueError(
                f"Could not align token '{token}' (index {idx}) in {source_path}."
            )
        end = start + len(token)
        spans.append({"start": start, "end": end})
        cursor = end
    return spans


def _sentence_to_row(
    *,
    text: str,
    tokens: list[str],
    labels: list[str],
    uid: int,
    source_path: Path,
) -> dict[str, Any]:
    token_spans = _align_tokens_to_text(text, tokens, source_path)
    bio = [
        {
            "start": span["start"],
            "end": span["end"],
            "text": text[span["start"] : span["end"]],
            "label": label,
        }
        for span, label in zip(token_spans, labels)
    ]
    return {"uid": uid, "text": text, "bio": bio}


def parse_conllu_file(path: Path, start_uid: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")

    rows: list[dict[str, Any]] = []
    uid = start_uid

    sentence_text: str | None = None
    tokens: list[str] = []
    labels: list[str] = []

    def flush_sentence() -> None:
        nonlocal uid, sentence_text, tokens, labels
        if not tokens:
            sentence_text = None
            return
        text = sentence_text if sentence_text is not None else " ".join(tokens)
        rows.append(
            _sentence_to_row(
                text=text,
                tokens=tokens,
                labels=labels,
                uid=uid,
                source_path=path,
            )
        )
        uid += 1
        sentence_text = None
        tokens = []
        labels = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line:
            flush_sentence()
            continue

        if line.startswith("#"):
            if line.startswith("# text = "):
                sentence_text = line[len("# text = ") :]
            continue

        cols = raw_line.split("\t")
        if len(cols) != 10:
            continue

        token_id = cols[0]
        if "-" in token_id or "." in token_id:
            # Skip multi-word tokens and empty nodes.
            continue

        token = cols[1]
        label = _parse_name_label(cols[9])
        tokens.append(token)
        labels.append(label)

    flush_sentence()
    return rows, uid


def build_dataset_dict(raw_dir: Path) -> DatasetDict:
    split_files = {
        "train": raw_dir / "ddt.train.conllu",
        "dev": raw_dir / "ddt.dev.conllu",
        "test": raw_dir / "ddt.test.conllu",
    }

    features = Features(
        {
            "uid": Value("int64"),
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

    uid_counter = 0
    split_datasets: dict[str, Dataset] = {}
    for split, file_path in split_files.items():
        rows, uid_counter = parse_conllu_file(file_path, start_uid=uid_counter)
        split_datasets[split] = Dataset.from_list(rows, features=features)

    return DatasetDict(split_datasets)


def copy_source_metadata_files(
    *,
    api: HfApi,
    source_repo_id: str,
    target_repo_id: str,
    token: str,
) -> None:
    source_files = api.list_repo_files(repo_id=source_repo_id, repo_type="dataset")
    # Do not copy dataset script files (*.py). Their presence causes
    # datasets.load_dataset(repo_id) to switch to script mode, which is disabled
    # in modern `datasets` releases.
    metadata_files = [
        x
        for x in source_files
        if not x.startswith("data/") and x != ".gitattributes" and not x.endswith(".py")
    ]

    for metadata_file in metadata_files:
        local_path = hf_hub_download(
            repo_id=source_repo_id,
            repo_type="dataset",
            filename=metadata_file,
            token=token,
        )

        path_to_upload = local_path
        if metadata_file == "README.md":
            path_to_upload = sanitize_readme_dataset_info(local_path)

        api.upload_file(
            path_or_fileobj=path_to_upload,
            path_in_repo=metadata_file,
            repo_id=target_repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Copy metadata file from {source_repo_id}: {metadata_file}",
        )


def sanitize_readme_dataset_info(readme_path: str) -> str:
    """Remove dataset_info frontmatter from source README.

    Keeping the original dataset_info from alexandrainst/dane would force
    `datasets` to cast parquet columns to an incompatible schema.
    """
    content = Path(readme_path).read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        return readme_path

    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return readme_path

    frontmatter = parts[1]
    body = parts[2]

    filtered: list[str] = []
    skip_dataset_info = False
    for line in frontmatter.splitlines():
        if not skip_dataset_info and line.startswith("dataset_info:"):
            skip_dataset_info = True
            continue

        if skip_dataset_info:
            # dataset_info is a top-level YAML key. Stop skipping on next
            # top-level key.
            if line and not line.startswith((" ", "\t")):
                skip_dataset_info = False
            else:
                continue

        if not skip_dataset_info:
            filtered.append(line)

    new_frontmatter = "\n".join(filtered).strip()
    rebuilt = f"---\n{new_frontmatter}\n---\n{body.lstrip()}"

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    tmp.write(rebuilt)
    tmp.flush()
    tmp.close()
    return tmp.name


def delete_target_loader_scripts(
    *, api: HfApi, target_repo_id: str, token: str
) -> None:
    target_files = api.list_repo_files(repo_id=target_repo_id, repo_type="dataset")
    loader_scripts = [x for x in target_files if x.endswith(".py") and "/" not in x]

    for script_file in loader_scripts:
        api.delete_file(
            path_in_repo=script_file,
            repo_id=target_repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Remove loader script {script_file} to enable parquet loading",
        )


def main() -> None:
    args = parse_args()
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise SystemExit("Missing HF_TOKEN. Set it in shell or .env before running.")

    ds_dict = build_dataset_dict(args.raw_dir)
    split_summary = ", ".join(
        f"{split}={len(ds_dict[split])}" for split in ds_dict.keys()
    )
    print(f"Built DANE dataset: {split_summary}")

    if args.dry_run:
        print("Dry run complete. Skipping upload.")
        return

    is_private = args.private
    print(
        f"Uploading dataset to {args.target_repo_id} "
        f"(private={is_private}, max_shard_size={args.max_shard_size}) ..."
    )
    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=args.target_repo_id,
        repo_type="dataset",
        private=is_private,
        exist_ok=True,
    )

    # Clean up stale dataset scripts from previous uploads.
    delete_target_loader_scripts(
        api=api, target_repo_id=args.target_repo_id, token=hf_token
    )

    ds_dict.push_to_hub(
        repo_id=args.target_repo_id,
        token=hf_token,
        private=is_private,
        max_shard_size=args.max_shard_size,
    )

    if not args.skip_metadata_copy:
        print(
            f"Copying metadata files from {args.source_repo_id} "
            f"to {args.target_repo_id} ..."
        )
        copy_source_metadata_files(
            api=api,
            source_repo_id=args.source_repo_id,
            target_repo_id=args.target_repo_id,
            token=hf_token,
        )

    # Ensure no loader scripts remain after metadata copy.
    delete_target_loader_scripts(
        api=api, target_repo_id=args.target_repo_id, token=hf_token
    )

    print("Upload complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
