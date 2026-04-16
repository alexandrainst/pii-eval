from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int


@dataclass
class LabelSpan(Span):
    label: str


def spans_to_bio(label_spans: list[LabelSpan], token_spans: list[Span]) -> list[str]:
    if not token_spans:
        return []

    sorted_labels = sorted(label_spans, key=lambda span: span.start)
    tags: list[str] = []
    label_idx = 0
    current = sorted_labels[label_idx] if sorted_labels else None

    for token in token_spans:
        while current is not None and token.start >= current.end:
            label_idx += 1
            if label_idx >= len(sorted_labels):
                current = None
                break
            current = sorted_labels[label_idx]

        if current is None or token.end <= current.start:
            tags.append("O")
            continue

        prefix = "B" if token.start <= current.start else "I"
        tags.append(f"{prefix}-{current.label}")

    return tags
