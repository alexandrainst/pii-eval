"""Helpers for BIO -> span conversion, char-offset reconstruction, label mapping and metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

try:
    # Use the project's existing label maps.
    from dfm_anonymization.constants import (  # type: ignore
        DATASET_LABEL_MAP,
        MODEL_LABEL_MAP,
        get_dataset_label_map,
        get_model_label_map,
    )
except Exception:  # pragma: no cover - fallback when project import fails
    DATASET_LABEL_MAP: dict[str, dict[str, str]] = {}
    MODEL_LABEL_MAP: dict[str, dict[str, str]] = {}

    def get_model_label_map(_name: str, default=None):
        return default

    def get_dataset_label_map(_name: str, default=None):
        return default


# Sentinel "key" the API uses to mean "do not apply any mapping (identity)".
NO_MAP_KEY = "__none__"


def resolve_default_model_map_key(model_name: str | None) -> str | None:
    """Return the regex key in MODEL_LABEL_MAP that ``get_model_label_map`` would resolve to.

    Returns ``None`` if no mapping matches.
    """
    if not model_name:
        return None
    import re

    for key in MODEL_LABEL_MAP:
        if re.compile(key).match(model_name):
            return key
    return None


def resolve_default_dataset_map_key(dataset_name: str | None) -> str | None:
    if not dataset_name:
        return None
    import re

    for key in DATASET_LABEL_MAP:
        if re.compile(key).match(dataset_name):
            return key
    return None

    def get_dataset_label_map(_name: str, default=None):
        return default


@dataclass
class Span:
    token_start: int
    token_end: int  # exclusive
    char_start: int
    char_end: int  # exclusive
    raw_label: str
    iso_label: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def bio_to_spans(
    bio_tags: list[str],
    tokens: list[str],
    token_spans: list[tuple[int, int]],
    text: str,
    label_map: dict[str, str] | None,
    drop_unmapped: bool = False,
) -> list[Span]:
    """Collapse a list of BIO tags into spans, applying ``label_map`` to obtain the canonical label.

    Tags lacking a B- prefix but starting an entity (defensive) are still treated as the start.

    If ``drop_unmapped`` is True and ``label_map`` is non-empty, spans whose raw label is not a key
    in ``label_map`` are discarded. If ``label_map`` is empty/None, the raw label is used as the
    iso label (identity).
    """
    spans: list[Span] = []
    if not bio_tags:
        return spans

    n = len(bio_tags)
    i = 0
    has_map = bool(label_map)
    label_map = label_map or {}

    while i < n:
        tag = bio_tags[i]
        if tag == "O" or tag is None:
            i += 1
            continue
        # B- or I- (treat I- starting a new entity as B-)
        if "-" in tag:
            _, raw_label = tag.split("-", 1)
        else:
            raw_label = tag
        start = i
        i += 1
        while i < n:
            nxt = bio_tags[i]
            if nxt == "O" or nxt is None:
                break
            if "-" in nxt:
                prefix, nxt_label = nxt.split("-", 1)
            else:
                prefix, nxt_label = "B", nxt
            if prefix == "I" and nxt_label == raw_label:
                i += 1
                continue
            break
        end = i

        if has_map:
            if raw_label not in label_map:
                if drop_unmapped:
                    continue
                iso = raw_label
            else:
                iso = label_map[raw_label]
        else:
            iso = raw_label

        cs = token_spans[start][0]
        ce = token_spans[end - 1][1]
        spans.append(
            Span(
                token_start=start,
                token_end=end,
                char_start=cs,
                char_end=ce,
                raw_label=raw_label,
                iso_label=iso,
                text=text[cs:ce],
            )
        )
    return spans


def reconstruct_token_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Find char offsets of ``tokens`` inside ``text`` by scanning left-to-right.

    Tokens are matched literally; whitespace between tokens is skipped. If a token can't be
    located, a zero-width span at the current cursor is recorded so indexing remains aligned.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    n = len(text)
    for tok in tokens:
        # If the token is itself whitespace (e.g. "\n"), match it literally.
        if tok and not tok.strip():
            # advance cursor to first occurrence of tok at/after current position
            idx = text.find(tok, cursor)
            if idx == -1:
                spans.append((cursor, cursor))
                continue
            spans.append((idx, idx + len(tok)))
            cursor = idx + len(tok)
            continue

        # Skip whitespace
        while cursor < n and text[cursor].isspace():
            cursor += 1
        if cursor >= n:
            spans.append((cursor, cursor))
            continue
        if text.startswith(tok, cursor):
            spans.append((cursor, cursor + len(tok)))
            cursor += len(tok)
        else:
            # Best-effort: search forward for the token within a small window.
            idx = text.find(tok, cursor)
            if idx == -1 or idx - cursor > 200:
                spans.append((cursor, cursor))
            else:
                spans.append((idx, idx + len(tok)))
                cursor = idx + len(tok)
    return spans


def map_labels_for_dataset(dataset_name: str) -> dict[str, str]:
    return get_dataset_label_map(dataset_name) or {}


def map_labels_for_model(model_name: str) -> dict[str, str]:
    return get_model_label_map(model_name) or {}


def resolve_label_map(
    kind: str,
    name: str | None,
    override_key: str | None,
) -> tuple[dict[str, str], str | None]:
    """Resolve a label map dict given an explicit override key.

    ``kind`` is ``"model"`` or ``"dataset"``.

    ``override_key`` semantics:
      * ``None``  -> auto: use the helper (regex match on ``name``).
      * ``NO_MAP_KEY`` (``"__none__"``) -> identity (no mapping).
      * any regex key present in the corresponding *_LABEL_MAP -> use that mapping verbatim.

    Returns ``(mapping_dict, resolved_key_or_None)``.
    """
    table = MODEL_LABEL_MAP if kind == "model" else DATASET_LABEL_MAP

    if override_key == NO_MAP_KEY:
        return {}, NO_MAP_KEY

    if override_key is None:
        # Auto: use helper-resolved key (if any).
        if kind == "model":
            resolved = resolve_default_model_map_key(name)
        else:
            resolved = resolve_default_dataset_map_key(name)
        return (table.get(resolved, {}) if resolved else {}), resolved

    # Explicit regex key from the table.
    if override_key in table:
        return table[override_key], override_key

    # Unknown key -> behave as no mapping (defensive).
    return {}, NO_MAP_KEY


def compute_metrics(
    gt_spans: list[Span], pred_spans: list[Span], n_tokens: int
) -> dict:
    """Compute span-level (exact match on token boundaries + iso label) and token-level metrics."""

    # Span-level (TP requires exact token boundaries AND identical iso_label)
    gt_set = {(s.token_start, s.token_end, s.iso_label) for s in gt_spans}
    pred_set = {(s.token_start, s.token_end, s.iso_label) for s in pred_spans}
    span_tp = len(gt_set & pred_set)
    span_fp = len(pred_set - gt_set)
    span_fn = len(gt_set - pred_set)

    def _safe(n, d):
        return (n / d) if d else 0.0

    span_p = _safe(span_tp, span_tp + span_fp)
    span_r = _safe(span_tp, span_tp + span_fn)
    span_f1 = _safe(2 * span_p * span_r, span_p + span_r)

    # Token-level (over canonical iso labels, "O" ignored for P/R)
    gt_tok = ["O"] * n_tokens
    pr_tok = ["O"] * n_tokens
    for s in gt_spans:
        for i in range(s.token_start, s.token_end):
            if 0 <= i < n_tokens:
                gt_tok[i] = s.iso_label
    for s in pred_spans:
        for i in range(s.token_start, s.token_end):
            if 0 <= i < n_tokens:
                pr_tok[i] = s.iso_label

    tp = fp = fn = 0
    for g, p in zip(gt_tok, pr_tok):
        if g == "O" and p == "O":
            continue
        if g == p:
            tp += 1
        elif g == "O":
            fp += 1
        elif p == "O":
            fn += 1
        else:
            fp += 1
            fn += 1
    tok_p = _safe(tp, tp + fp)
    tok_r = _safe(tp, tp + fn)
    tok_f1 = _safe(2 * tok_p * tok_r, tok_p + tok_r)

    return {
        "span": {
            "tp": span_tp,
            "fp": span_fp,
            "fn": span_fn,
            "precision": span_p,
            "recall": span_r,
            "f1": span_f1,
        },
        "token": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tok_p,
            "recall": tok_r,
            "f1": tok_f1,
        },
    }


def spans_to_dicts(spans: Iterable[Span]) -> list[dict]:
    return [s.to_dict() for s in spans]
