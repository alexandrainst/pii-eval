from dataclasses import dataclass
from typing import Callable

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


def spans_to_bio(
    text: str,
    spans: list[Span],
    tokenize: Callable[[str], list[TextSpan]] = WordTokenizer(),
) -> list[LabelSpan]:
    started_spans: set[int] = set()
    result = []
    for word in tokenize(text):
        label = "O"
        for i, span in enumerate(spans):
            if word.start >= span.start and word.end <= span.end:
                span_label = getattr(span, "label", "ENT")
                prefix = "I" if i in started_spans else "B"
                started_spans.add(i)
                label = f"{prefix}-{span_label}"
                break
        result.append(
            LabelSpan(start=word.start, end=word.end, text=word.text, label=label)
        )

    return result
