from collections import Counter
from collections.abc import Iterable, Set as AbcSet
from dataclasses import fields, is_dataclass
from typing import Any, Mapping


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def find_alnum(haystack: str, needle: str, cursor: int = 0) -> int:
    haystack = haystack[cursor:]
    # Strip non-alnum from needle
    n = "".join(c for c in needle if c.isalnum())
    if not n:
        return -1
    # Build cleaned haystack + map: cleaned index -> original index
    cleaned = []
    idx_map = []
    for i, c in enumerate(haystack):
        if c.isalnum():
            cleaned.append(c)
            idx_map.append(i)
    pos = "".join(cleaned).find(n)
    return idx_map[pos] + cursor if pos != -1 else -1


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


def normalize_bio_label(label: str) -> str:
    if label == "O":
        return "O"
    if label.startswith(("B-", "I-")):
        return label.split("-", 1)[1]
    return label


def count_labels_from_entries(entries: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for entry in entries:
        for token in entry.bio:
            counter[normalize_bio_label(token.label)] += 1
    return dict(sorted(counter.items()))


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
