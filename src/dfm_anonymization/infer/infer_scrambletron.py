import re
import json
from pathlib import Path
from typing import List, Tuple

from scrambletron.utils import create_analyzer
from tqdm import tqdm

from dfm_anonymization.util.misc import DATASET_LABEL_MAP

OVERRIDE = False
analyzer = create_analyzer(device="cuda")

for dataset_name in list(reversed(DATASET_LABEL_MAP.keys())):

    print(dataset_name)
    input_fp = Path(f"data/span/{dataset_name}.json").resolve()
    output_path = Path(f"data/infer/{dataset_name}/scrambletron.json").resolve()
    if output_path.exists() and not OVERRIDE:
        print("output exists, skipping")
        continue

    dataset: List[dict] = json.loads(input_fp.read_text(encoding="utf8"))

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
    for idx, document in tqdm(
        enumerate(dataset), total=len(dataset), dynamic_ncols=True
    ):
        text = document.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text_len = len(text)
        spans = []
        seen = set()

        for chunk_start, chunk_text in _chunk_text(text):
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
                        "text": text[start_idx:end_idx],
                    }
                )

        spans.sort(key=lambda span: span["start"])
        results.append(
            {
                "document": document.get("document", idx),
                "text": text,
                "spans": spans,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=4), encoding="utf8")
