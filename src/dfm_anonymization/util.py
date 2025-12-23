from collections.abc import Set as AbcSet
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Mapping, Set, Tuple

import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.tokenize.util import align_tokens
from presidio_anonymizer import RecognizerResult


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class WordTokenizer(metaclass=Singleton):
    def __init__(self):
        nltk.download("punkt")
        nltk.download("punkt_tab")

    def word_tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        words = word_tokenize(text, language="danish")
        word_spans = align_tokens(tokens=words, sentence=text)
        return words, word_spans


def tokens_to_offsets(text: str, tokens: List[str]) -> List[tuple[int, int]]:
    offsets: List[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if not token:
            offsets.append((cursor, cursor))
            continue
        start = text.find(token, cursor)
        if start == -1:
            raise ValueError(f"Token '{token}' not found in text segment.")
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def filter_bio(bio_tags: List[str], labels: Set[str]) -> List[str]:
    return [tag if tag[2:] in labels else "O" for tag in bio_tags]


def spans_to_bio(
    token_spans: List[tuple[int, int]], spans: List[Dict[str, Any]]
) -> List[str]:
    tags = ["O"] * len(token_spans)
    for span in sorted(spans, key=lambda item: (int(item["start"]), int(item["end"]))):
        start = int(span["start"])
        end = int(span["end"])
        label = span["label"]
        inside = False
        for idx, (tok_start, tok_end) in enumerate(token_spans):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            if tok_start < end and tok_end > start:
                prefix = "B" if not inside else "I"
                if tags[idx] == "O":
                    tags[idx] = f"{prefix}-{label}"
                inside = True
    return tags


def presidio_result_2_span(result: RecognizerResult, text: str) -> dict:
    return {
        "label": result.entity_type,
        "start": result.start,
        "end": result.end,
        "text": text[result.start : result.end],
    }


def get_torch_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else "cpu"
    )


def to_dict_recursive(
    obj: Any, *, include_private: bool = False, max_depth: int | None = None
) -> Any:
    """
    Recursively convert custom objects into JSON-like primitives (dict/list/etc).

    - Supports: dataclasses, objects with __dict__ / __slots__, mappings, iterables, sets/tuples
    - Avoids infinite recursion via cycle detection
    - Leaves primitives unchanged
    """
    seen: dict[int, Any] = {}

    def is_primitive(x: Any) -> bool:
        return x is None or isinstance(x, (bool, int, float, str, bytes))

    def convert(x: Any, depth: int) -> Any:
        if max_depth is not None and depth > max_depth:
            return repr(x)

        if is_primitive(x):
            return x

        if isinstance(x, (type,)):
            return x.__qualname__

        oid = id(x)
        if oid in seen:
            return {"__cycle__": True, "__type__": type(x).__qualname__}
        # Mark as seen early (handles self-referential graphs)
        seen[oid] = True

        # Mappings
        if isinstance(x, Mapping):
            out = {}
            for k, v in x.items():
                ck = convert(k, depth + 1)
                # dict keys must be hashable; coerce complex keys to strings
                try:
                    hash(ck)
                except Exception:
                    ck = repr(ck)
                out[ck] = convert(v, depth + 1)
            return out

        # Common iterables (but not strings/bytes which are primitives above)
        if isinstance(x, (list, tuple)):
            out = [convert(i, depth + 1) for i in x]
            return out if isinstance(x, list) else tuple(out)

        if isinstance(x, (set, frozenset, AbcSet)):
            return sorted((convert(i, depth + 1) for i in x), key=lambda y: repr(y))

        # Dataclasses
        if is_dataclass(x) and not isinstance(x, type):
            return {f.name: convert(getattr(x, f.name), depth + 1) for f in fields(x)}

        # Objects with __dict__ / __slots__
        data: dict[str, Any] = {}

        if hasattr(x, "__dict__"):
            for k, v in vars(x).items():
                if not include_private and k.startswith("_"):
                    continue
                data[k] = convert(v, depth + 1)

        slots = getattr(type(x), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for s in slots or ():
            if s in ("__dict__", "__weakref__"):
                continue
            if not include_private and str(s).startswith("_"):
                continue
            if hasattr(x, s):
                data[str(s)] = convert(getattr(x, s), depth + 1)

        if data:
            return data

        # Fallback for unknown types (e.g., datetime, Path, enums): string
        # representation
        return repr(x)

    return convert(obj, 0)
