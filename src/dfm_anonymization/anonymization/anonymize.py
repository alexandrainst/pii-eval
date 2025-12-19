from abc import ABC, abstractmethod
from typing import List, Tuple
import nltk
from nltk.tokenize import TreebankWordTokenizer
from presidio_anonymizer import AnonymizerEngine, EngineResult
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig

from dfm_anonymization.inference.dataset import PiiDataEntry
from dfm_anonymization.inference.gliner_model import GlinerModel
from dfm_anonymization.inference.interfaces import PiiModel
from dfm_anonymization.util.misc import LABEL_DESCRIPTIONS_DA


class Analyzer(ABC):
    @abstractmethod
    def __call__(self, text: str) -> List[RecognizerResult]:
        pass


def spans_from_bio_tags(
        bio_tags: List[str], spans: List[Tuple[int, int]]
) -> List[RecognizerResult]:
    if len(bio_tags) != len(spans):
        raise ValueError("bio_tags and spans must be of equal length")

    results: List[RecognizerResult] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def flush():
        nonlocal current_label, current_start, current_end
        if current_label is not None and current_start is not None and current_end is not None:
            results.append(
                RecognizerResult(
                    entity_type=current_label,
                    start=current_start,
                    end=current_end,
                    score=1.0
                )
            )
            current_label = None
            current_start = None
            current_end = None

    for tag, (start, end) in zip(bio_tags, spans):
        if not tag or tag == "O":
            flush()
            continue

        prefix, sep, label = tag.partition("-")
        if not sep:
            label = prefix
            prefix = "B"

        if prefix == "B" or current_label != label:
            flush()
            current_label = label
            current_start = start

        current_end = end

    flush()
    return results


class PiiModelAnalyzer(Analyzer):
    def __init__(self, model: PiiModel = GlinerModel()):
        super().__init__()
        self.model = model
        nltk.download('punkt')
        self.word_tokenizer = TreebankWordTokenizer()

    def __call__(self, text: str) -> List[RecognizerResult]:
        word_spans = list(self.word_tokenizer.span_tokenize(text))
        words = [text[s:e] for s, e in word_spans]
        bio_tags = self.model(PiiDataEntry(text=text, tokens=words))
        return spans_from_bio_tags(bio_tags=bio_tags, spans=word_spans)


class Anonymizer:
    def __init__(self, analyzer: Analyzer = PiiModelAnalyzer()):
        self.engine = AnonymizerEngine()
        self.analyzer = analyzer
        self.operators = {
            k: OperatorConfig("replace", {"new_value": f"<{k}>"})
            for k in LABEL_DESCRIPTIONS_DA.keys()
        }

    def __call__(self, text: str) -> EngineResult:
        return self.engine.anonymize(
            text=text,
            analyzer_results=self.analyzer(text),
            operators=self.operators
        )


if __name__ == "__main__":
    # text = "Navnet er Sand, Anders Sand. Og 21324354 er telefonnummeret."
    text = "Marias kreditkortnummer, 4929 1234 5678 9012, blev ved et uheld delt i en e-mail til supporten."
    anonymizer = Anonymizer()
    anonymized = anonymizer(text)
    print(text)
    print(anonymized.text)
