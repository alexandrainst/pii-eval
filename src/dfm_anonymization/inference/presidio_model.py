
from typing import List
import spacy
from dfm_anonymization.inference.dataset import PiiDataEntry
from dfm_anonymization.inference.interfaces import PiiModel
from dfm_anonymization.util.util import get_torch_device, presidio_result_2_span, spans_to_bio, tokens_to_offsets
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider


class PresidioModel(PiiModel):
    def __init__(
            self, name: str = "presidio",
            lang_code: str = "da",
            model_name="da_core_news_trf"
    ):
        super().__init__(name)
        self.lang_code = lang_code
        self.configuration = {
            "nlp_engine_name": "spacy",
            "models": [{
                "lang_code": self.lang_code,
                "model_name": model_name
            }],
        }
        if get_torch_device() in ["cuda", "xpu"]:
            spacy.require_gpu(0)
        provider = NlpEngineProvider(nlp_configuration=self.configuration)
        nlp_engine_da = provider.create_engine()

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine_da,
            supported_languages=[self.lang_code]
        )

    def __call__(self, entry: PiiDataEntry) -> List[str]:
        results = self.analyzer.analyze(
            entry.text, language=self.lang_code)
        spans = [presidio_result_2_span(r, entry.text) for r in results]
        token_spans = tokens_to_offsets(entry.text, tokens=entry.tokens)
        return spans_to_bio(token_spans=token_spans, spans=spans)


