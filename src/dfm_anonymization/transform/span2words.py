# TODO (ACTED ON 2025-12-12T11:49:57+00:00): this script is supposed to take all json files of the pattern <repos_root>/data/span/*.json
# and convert them - saving each file to <repos_root>/data/words/<filename>.json
# the converted file contains json data with a similar structure to the original file but with for each top level list item:
# - the 'spans' item is removed and replace by these other items instead:
# - a list of tokens output by `nltk.tokenize.sent_tokenize(text, language='danish')`
# - a list with same size containing BIO-tags generated from the spans (B-<label>, I-<label>, O, etc.) where label name is taken from the span
# - also add a string key + value "language": "danish" for all items (they are all in danish in our case)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import nltk
from nltk.tokenize import TreebankWordTokenizer


SentenceSpans = Sequence[Tuple[int, int]]
SpanEntry = Dict[str, Any]
ExampleEntry = Dict[str, Any]

nltk.download('punkt')
_TOKENIZER = TreebankWordTokenizer()


def _tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    spans = list(_TOKENIZER.span_tokenize(text))
    if not spans and text:
        spans = [(0, len(text))]
    tokens = [text[start:end] for start, end in spans]
    return tokens, spans


def _spans_to_bio(token_spans: SentenceSpans, spans: Sequence[SpanEntry]) -> List[str]:
    tags = ["O"] * len(token_spans)
    ordered_spans = sorted(spans, key=lambda span: (span["start"], span["end"]))
    for span in ordered_spans:
        start = int(span["start"])
        end = int(span["end"])
        label = span["label"]
        inside = False
        for idx, (tok_start, tok_end) in enumerate(token_spans):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            overlaps = tok_start < end and tok_end > start
            if overlaps:
                prefix = "B" if not inside else "I"
                tag = f"{prefix}-{label}"
                if tags[idx] == "O":
                    tags[idx] = tag
                inside = True
    return tags


def _transform_entry(entry: ExampleEntry) -> ExampleEntry:
    text = entry.get("text", "")
    spans = entry.get("spans", [])
    tokens, token_spans = _tokenize_with_spans(text)
    tags = _spans_to_bio(token_spans, spans)
    result = dict(entry)
    result.pop("spans", None)
    result["words"] = tokens
    result["bio_tags"] = tags
    return result


def _convert_file(source: Path, destination_dir: Path) -> None:
    with source.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)
    transformed = [_transform_entry(entry) for entry in entries]
    destination = destination_dir / source.name
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(transformed, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _convert_all(repo_root: Path) -> None:
    span_dir = repo_root / "data" / "span"
    words_dir = repo_root / "data" / "words"
    words_dir.mkdir(parents=True, exist_ok=True)
    for path in span_dir.glob("*.json"):
        if path.is_file():
            _convert_file(path, words_dir)


def main() -> None:
    _convert_all(Path("."))


if __name__ == "__main__":
    main()
