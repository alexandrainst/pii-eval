from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable

import spacy
import spacy.cli

from pii_eval.util import Singleton


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

    @staticmethod
    def from_text_and_tokens(
        text: str,
        tokens: list[str],
        labels: list[str] | None = None,
        include_o_labels: bool = True,
    ) -> list["LabelSpan"]:
        if labels is None:
            labels = ["O" for _ in tokens]
        spans = []
        last_end = 0
        for token, label in zip(tokens, labels):
            start = text.find(token, last_end)
            end = start + len(token)
            last_end = end
            if include_o_labels or label != "O":
                spans.append(
                    LabelSpan(start=start, end=end, text=text[start:end], label=label)
                )
        return spans


class Tokenizer(ABC):
    @abstractmethod
    def __call__(self, text: str) -> list[TextSpan]: ...


class WordTokenizer(Tokenizer, metaclass=Singleton):
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


def align_bio(
    text: str, bio: list[LabelSpan], tokenizer: Tokenizer = WordTokenizer()
) -> list[LabelSpan]:
    tokens = tokenizer(text)
    copy = bio.copy()
    current = copy.pop() if copy else None
    result = []
    for token in tokens:
        label = "O"
        if current:
            if current.start <= token.start:
                label = current.label
                if current.end <= token.end:
                    current = copy.pop() if copy else None
        result.append(LabelSpan(**asdict(token), label=label))
    return result


def map_bio(spans: list[LabelSpan], label_map: dict[str, str]) -> list[LabelSpan]:
    mapped: list[LabelSpan] = []
    append = mapped.append

    for span in spans:
        new_span = deepcopy(span)
        if span == "O" or "-" not in span:
            new_span.label = "O" if span == "" else span
        else:
            prefix, label = span.split("-", 1)
            mapped_label = label_map.get(label)
            if mapped_label is None:
                new_span.label = "O"
            else:
                append(f"{prefix}-{mapped_label}")
            mapped.append(new_span)

    return mapped


def spans_to_bio(
    text: str,
    spans: list[LabelSpan],
    tokenize: Callable[[str], list[TextSpan]] = WordTokenizer(),
    include_o_labels: bool = True,
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
        if include_o_labels or label != "O":
            result.append(LabelSpan(**asdict(token), label=label))
    return result


def bio_to_spans(text: str, bio: list[LabelSpan]) -> list[LabelSpan]:
    spans = []
    current_span_start = None
    current_span_end = None
    current_label = None

    for token in bio:
        if token.label.startswith("B-"):
            # Start a new span
            if current_span_start is not None:
                span_text = text[current_span_start:current_span_end]
                spans.append(
                    LabelSpan(
                        start=current_span_start,
                        end=current_span_end,
                        text=span_text,
                        label=current_label,
                    )
                )
            label = token.label[2:]
            current_label = label
            current_span_start = token.start
            current_span_end = token.end
        elif token.label.startswith("I-"):
            # Continue current span
            label = token.label[2:]
            if current_span_start is not None and current_label == label:
                current_span_end = token.end
        elif token.label == "O":
            # Outside any span
            if current_span_start is not None:
                span_text = text[current_span_start:current_span_end]
                spans.append(
                    LabelSpan(
                        start=current_span_start,
                        end=current_span_end,
                        text=span_text,
                        label=current_label,
                    )
                )
                current_span_start = None
                current_span_end = None
                current_label = None

    # Close any remaining span
    if current_span_start is not None:
        span_text = text[current_span_start:current_span_end]
        spans.append(
            LabelSpan(
                start=current_span_start,
                end=current_span_end,
                text=span_text,
                label=current_label,
            )
        )

    return spans


# Example of input LabelSpan texts:
# ['Virksomhedens', 'interne', 'profil', 'for', 'medarbejdere', 'ligger', 'på', 'https://intra.firma.local/employee/123', 'hvor', 'brugernavnet', 'er', 'mhansen', '.', 'På', 'intranettet', 'findes', 'også', 'adgangsloggene', '(', 'sidst', 'set', 'fra', 'IP', '192.168.0.1', ')', '.']
# ['Virksomhedens interne', 'profil', 'for', 'medarbejdere', 'ligger', 'på', 'https', ':', '/', '/', 'intra', '.', 'firma', '.', 'local', '/', 'employee', '/', '123', 'hvor', 'brugernavnet', 'er', 'mhansen', '.', 'På', 'intranettet', 'findes', 'også', 'adgangsloggene', '(', 'sidst', 'set', 'fra', 'IP', '192', '.', '168', '.', '0', '.', '1', ')', '.']
# Example of "aligning":
# ['Virksomhedens', ' ', 'interne', 'profil', ...., 'på', 'https', ':', ...., '123', 'hvor', ...]

# In cases where a label does not exist for a 'token' for either 'left' or 'right' label should be 'O'


#     boundaries = sorted(
#         {pos for span in left + right for pos in (span.start, span.end)}
#     )
#
#     def find_span(spans: list[LabelSpan], start: int, end: int) -> LabelSpan | None:
#         for span in spans:
#             if span.start <= start and end <= span.end:
#                 return span
#         return None
#
#     def make_sub_span(
#         span: LabelSpan, start: int, end: int, started: set[int]
#     ) -> LabelSpan:
#         text = span.text[start - span.start : end - span.start]
#         label = (
#             "I-" + span.label[2:]
#             if span.label.startswith("B-") and id(span) in started
#             else span.label
#         )
#         started.add(id(span))
#         return LabelSpan(start=start, end=end, text=text, label=label)
#
#     def make_o(start: int, end: int, other: LabelSpan | None) -> LabelSpan:
#         text = other.text[start - other.start : end - other.start] if other else ""
#         return LabelSpan(start=start, end=end, text=text, label="O")
#
#     left_started: set[int] = set()
#     right_started: set[int] = set()
#     result = []
#     for i in range(len(boundaries) - 1):
#         s, e = boundaries[i], boundaries[i + 1]
#         l_span = find_span(left, s, e)
#         r_span = find_span(right, s, e)
#         l = (
#             make_sub_span(l_span, s, e, left_started)
#             if l_span is not None
#             else make_o(s, e, r_span)
#         )
#         r = (
#             make_sub_span(r_span, s, e, right_started)
#             if r_span is not None
#             else make_o(s, e, l_span)
#         )
#         if l.text != r.text:
#             raise ValueError(f"Text mismatch at [{s}, {e}]: {l.text!r} != {r.text!r}")
#         if l.text:
#             result.append((l, r))
#     return result
