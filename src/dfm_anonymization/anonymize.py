import gzip
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

from presidio_anonymizer import AnonymizerEngine, EngineResult
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

from dfm_anonymization.constants import LABEL_DESCRIPTIONS_DA
from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.models.ensemble_model import EnsembleModel
from dfm_anonymization.models.gliner_model import GlinerModel
from dfm_anonymization.util.util import WordTokenizer


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
        if (
            current_label is not None
            and current_start is not None
            and current_end is not None
        ):
            results.append(
                RecognizerResult(
                    entity_type=current_label,
                    start=current_start,
                    end=current_end,
                    score=1.0,
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


class Analyzer(ABC):
    @abstractmethod
    def __call__(self, text: str) -> List[RecognizerResult]:
        pass


class PiiModelAnalyzer(Analyzer):
    def __init__(self, model: PiiModel | None = None):
        super().__init__()
        self.model = model if model else EnsembleModel.get_gliner_pii_presidio()

    def __call__(self, text: str) -> List[RecognizerResult]:
        words, word_spans = WordTokenizer().word_tokenize(text)
        bio_tags = self.model(PiiDataEntry(text=text, tokens=words))
        return spans_from_bio_tags(bio_tags=bio_tags, spans=word_spans)


class WikiDa:
    def __init__(self):
        data_path = Path(".cache/dawiki-latest-all-titles-in-ns0")
        url = "https://dumps.wikimedia.org/dawiki/latest/dawiki-latest-all-titles-in-ns0.gz"
        if not data_path.exists():
            data_path.parent.mkdir(exist_ok=True, parents=True)
            compressed_path = f"{data_path}.gz"
            urlretrieve(url=url, filename=compressed_path)
            with gzip.open(compressed_path, "rb") as f_in:
                with open(data_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        content = data_path.read_text(encoding="utf8")
        lines = content.split("\n")
        title_set = set(line.replace("_", " ").replace("\n", "") for line in lines)
        self._title_set = set(t for t in title_set if len(t) > 3)

    def look_up(self, key: str) -> bool:
        return key in self._title_set


class Anonymizer:
    def __init__(self, analyzer: Analyzer | None = None):
        self.engine = AnonymizerEngine()
        self.wiki = WikiDa()
        self.analyzer = analyzer if analyzer else PiiModelAnalyzer()
        self.operators = {
            k: OperatorConfig("replace", {"new_value": f"<{k}>"})
            for k in LABEL_DESCRIPTIONS_DA.keys()
        }

    def __call__(
        self, text: str, whitelist: Optional[List[str]] = None
    ) -> EngineResult:
        # ignore wiki entities and whitelist entries
        analyser_results = []
        for r in self.analyzer(text):
            substr = text[r.start : r.end]
            if not self.wiki.look_up(substr) and (
                not whitelist or substr not in whitelist
            ):
                analyser_results.append(r)

        return self.engine.anonymize(
            text=text, analyzer_results=analyser_results, operators=self.operators
        )


if __name__ == "__main__":
    text = (
        "Navnet er Anders Sand. Og 21324354 er telefonnummeret. "
        "Marias kreditkortnummer, 4929 1234 5678 9012, blev ved et uheld "
        "delt i en e-mail til supporten. Peter Lund Madsen er død."
    )
    anonymizer = Anonymizer(
        Analyzer(PiiModelAnalyzer(model=GlinerModel(tag="nvidia/gliner-pii")))
    )
    anonymized = anonymizer(text)
    print(text)
    print(anonymized.text)
