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
    # TODO implement this
    raise NotImplemented


class PiiModelAnalyzer(Analyzer):
    def __init__(self, model: PiiModel = GlinerModel()):
        super().__init__()
        self.model = model
        nltk.download('punkt')
        self.word_tokenizer = TreebankWordTokenizer()

    def __call__(self, text: str) -> List[RecognizerResult]:
        word_spans = self.word_tokenizer.span_tokenize(text)
        words = [text[s:e] for e, s in word_spans]
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
    text = "My name is Bond, James Bond"
    anonymizer = Anonymizer()
    anonymized = anonymizer(text)
    print(anonymized)
