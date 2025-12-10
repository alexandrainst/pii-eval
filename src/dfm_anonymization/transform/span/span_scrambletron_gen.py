from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

RAW_DIR = Path("data/raw/scrambletron_gen").resolve()
OUTPUT_DIR = Path("data/span/scrambletron_gen").resolve()


def _derive_label(path: Path) -> str:
    name = path.stem
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def _locate_span(
    text: str,
    entity: str,
    occupied: List[Tuple[int, int]],
    context: str,
) -> Optional[Tuple[int, int]]:
    search_start = 0
    while True:
        start = text.find(entity, search_start)
        if start == -1:
            return None
        end = start + len(entity)
        for occ_start, occ_end in occupied:
            if start < occ_end and end > occ_start:
                break
        else:
            return start, end
        search_start = start + 1


def _convert_file(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Unexpected data layout in {path}")

    label = _derive_label(path)
    converted: List[dict] = []

    skipped = 0
    for idx, sample in enumerate(samples):
        sentence = sample.get("sentence")
        if not isinstance(sentence, str):
            raise ValueError(f"Sample {idx} missing sentence in {path.name}")
        entity_texts = sample.get("entities") or []
        if not isinstance(entity_texts, list):
            raise ValueError(
                f"Sample {idx} has invalid entities in {path.name}"
            )

        spans = []
        occupied: List[Tuple[int, int]] = []
        for entity in entity_texts:
            if not isinstance(entity, str):
                raise ValueError(
                    f"Entity {entity!r} is not text in {path.name}"
                )
            optional_span = _locate_span(
                sentence, entity, occupied, f"{path.name} sample #{idx}"
            )
            if optional_span:
                start, end = optional_span
            else:
                skipped += 1
                continue
            occupied.append((start, end))
            spans.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": sentence[start:end],
                }
            )

        spans.sort(key=lambda span: (span["start"], span["end"]))
        converted.append(
            {
                "document": idx,
                "full_text": sentence,
                "spans": spans,
            }
        )

    return converted, skipped


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Input directory missing: {RAW_DIR}")

    for input_file in sorted(RAW_DIR.glob("*.json")):
        converted, skipped = _convert_file(input_file)
        if len(converted) == 0:
            continue
        out_path = OUTPUT_DIR / input_file.name
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(converted, handle, ensure_ascii=False, indent=2)
        print(f"Wrote {len(converted)} documents to {out_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
