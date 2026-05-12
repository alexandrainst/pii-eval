from copy import deepcopy
from dataclasses import dataclass
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
    def from_text_and_token(
        text: str, tokens: list[str], labels: list[str]
    ) -> list["LabelSpan"]:
        spans = []
        last_end = 0
        for token, label in zip(tokens, labels):
            start = text.find(token, last_end)
            end = start + len(token)
            last_end = end
            spans.append(
                LabelSpan(start=start, end=end, text=text[start:end], label=label)
            )
        return spans


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

def align_bios(
    left: list[LabelSpan], right: list[LabelSpan]
) -> list[tuple[LabelSpan, LabelSpan]]:
    raise NotImplementedError
