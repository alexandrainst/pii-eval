import re
import json
from pathlib import Path
from typing import List, Tuple

from scrambletron.utils import create_analyzer
from tqdm import tqdm

data_fp = Path("data/processed/span/reduced/piilal.json").resolve()
with open(data_fp, "r") as f:
    ds: List[dict] = json.load(f)

analyzer = create_analyzer()


def _chunk_text(text: str) -> List[Tuple[int, str]]:
    if not text:
        return []
    chunks: List[Tuple[int, str]] = []
    start = 0
    for match in re.finditer(r"[.\n]", text):
        end = match.end()
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((start, chunk))
        start = end
    if start < len(text):
        chunk = text[start:]
        if chunk.strip():
            chunks.append((start, chunk))
    if not chunks and text.strip():
        chunks.append((0, text))
    return chunks


results = []
for idx, document in tqdm(enumerate(ds), total=len(ds)):
    full_text = document.get("full_text", "")
    if not isinstance(full_text, str):
        full_text = str(full_text)
    text_len = len(full_text)
    spans = []
    seen = set()

    for chunk_start, chunk_text in _chunk_text(full_text):
        analysis = analyzer.analyze(chunk_text, language="da")
        for item in analysis:
            start_local = getattr(item, "start", None)
            end_local = getattr(item, "end", None)
            label = getattr(item, "entity_type", None)
            if start_local is None or end_local is None or label is None:
                continue
            try:
                start_idx = chunk_start + int(start_local)
                end_idx = chunk_start + int(end_local)
            except (TypeError, ValueError):
                continue
            if start_idx < 0:
                start_idx = 0
            if end_idx < 0:
                end_idx = 0
            if start_idx > text_len:
                start_idx = text_len
            if end_idx > text_len:
                end_idx = text_len
            if end_idx <= start_idx:
                continue
            key = (label, start_idx, end_idx)
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                {
                    "label": label,
                    "start": start_idx,
                    "end": end_idx,
                    "text": full_text[start_idx:end_idx],
                }
            )

    spans.sort(key=lambda span: span["start"])
    results.append(
        {
            "document": document.get("document", idx),
            "full_text": full_text,
            "spans": spans,
        }
    )

output_path = Path("data/infer/pii_lal/scrambletron.json").resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2)

reduced_path = Path(
    "data/infer/pii_lal/reduced/scrambletron_red.json").resolve()
reduced_path.parent.mkdir(parents=True, exist_ok=True)
with open(reduced_path, "w") as handle:
    json.dump(results[:5], handle, ensure_ascii=False, indent=2)
