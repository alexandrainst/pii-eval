from collections.abc import Set as AbcSet
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Mapping

import spacy
import spacy.cli


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class Span:
    start: int
    end: int


@dataclass
class TextSpan(Span):
    text: str


@dataclass
class LabelSpan(TextSpan):
    label: str


class WordTokenizer(metaclass=Singleton):
    def __init__(self):
        try:
            self.nlp = spacy.load("da_core_news_sm")
        except OSError:
            spacy.cli.download("da_core_news_sm")
            self.nlp = spacy.load("da_core_news_sm")

    def __call__(self, text: str) -> list[TextSpan]:
        return [
            TextSpan(text=word.text, start=word.idx, end=word.idx + len(word.text))
            for word in self.nlp(text)
        ]


def autoinit(cls):
    fields = list(cls.__annotations__.items())

    def __init__(self, *args, **kwargs):
        for (name, _), val in zip(fields, args):
            setattr(self, name, val)
        for name, _ in fields:
            if name not in [f for f, _ in zip(fields, args)]:
                setattr(self, name, kwargs[name])

    cls.__init__ = __init__
    return cls


def spans_to_bio(
    text: str,
    spans: list[LabelSpan],
    tokenize: Callable[[str], list[TextSpan]] = WordTokenizer(),
) -> list[LabelSpan]:
    started_spans: set[int] = set()
    result = []
    for token in tokenize(text):
        label = "O"
        for i, span in enumerate(spans):
            if token.start >= span.start and token.end <= span.end:
                prefix = "I" if i in started_spans else "B"
                started_spans.add(i)
                label = f"{prefix}-{span.label}"
                break
        result.append(
            LabelSpan(start=token.start, end=token.end, text=token.text, label=label)
        )

    return result


def bio_to_spans(text: str, bio: list[LabelSpan]) -> list[LabelSpan]:

    raise NotImplementedError


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
