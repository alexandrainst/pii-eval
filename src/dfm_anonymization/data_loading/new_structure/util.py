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

    def __call__(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        words = []
        spans = []
        doc = self.nlp(text)
        for word in doc:
            words.append(word.text)
            spans.append((word.idx, word.idx + len(word.text)))
        return words, spans


def spans_to_bio(
    text: str,
    spans: list[Span],
    tokenize: Callable[[str], list[str]] = WordTokenizer(),
) -> list[LabelSpan]:
    tokens = tokenize(text)
    raise NotImplementedError()  # TODO
