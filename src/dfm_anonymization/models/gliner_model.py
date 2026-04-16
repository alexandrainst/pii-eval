from collections.abc import Set as AbcSet
from typing import Any, Dict, List, Set

from bidict import bidict

from dfm_anonymization.constants import LABEL_DESCRIPTIONS_DA
from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.util.util import (
    filter_bio,
    get_torch_device,
    spans_to_bio,
    tokens_to_offsets,
)

GLINER_MODEL_EXAMPLES = ["nvidia/gliner-pii", "urchade/gliner_multi-v2.1"]


class GlinerModel(PiiModel):
    def __init__(
        self,
        tag: str = GLINER_MODEL_EXAMPLES[0],
        labels: Set[str] | bidict[str, str] = LABEL_DESCRIPTIONS_DA,
    ):
        super().__init__(name=tag.split("/")[-1])
        self.tag = tag
        self.model: Any | None = None
        self.labels = labels

    def _load_model(self) -> Any:
        try:
            from gliner import GLiNER
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import GLiNER for model '{self.tag}'. "
                "The installed gliner/transformers/torch stack appears incompatible "
                "with the current Python runtime. This commonly fails on Python 3.13; "
                "try a Python 3.12 environment or a dependency combination known to "
                "support your interpreter."
            ) from exc
        return GLiNER.from_pretrained(self.tag, map_location=get_torch_device())

    def __call__(
        self, entry: PiiDataEntry, label_whitelist: Set[str] | None = None
    ) -> List[str]:
        if self.model is None:
            self.model = self._load_model()
        descriptions = (
            list(self.labels)
            if isinstance(self.labels, AbcSet)
            else list(self.labels.values())
        )
        token_spans = entry.get_token_spans() or tokens_to_offsets(entry.text, entry.tokens)
        max_sentence_len = 500
        c = 0
        subtexts = []
        span_inits = []
        for tspan in token_spans:
            end = tspan[1]
            if end - c > max_sentence_len:
                subtexts.append(entry.text[c:end])
                span_inits.append(c)
                c = end
        if c < len(entry.text) - 1:
            subtexts.append(entry.text[c:])
            span_inits.append(c)
        subspan_list: List[List[Dict[str, Any]]] = self.model.batch_predict_entities(
            subtexts, descriptions
        )
        spans = []
        for c, subspans in zip(span_inits, subspan_list):
            for span in subspans:
                span["start"] += c
                span["end"] += c
                spans.append(span)
        if isinstance(self.labels, bidict):
            for span in spans:
                span["label"] = self.labels.inverse[span["label"]]
        bio_tags = spans_to_bio(token_spans, spans)
        return filter_bio(bio_tags, label_whitelist) if label_whitelist else bio_tags


if __name__ == "__main__":
    gliner = GlinerModel(GLINER_MODEL_EXAMPLES[0], bidict({"n": "navn", "m": "verbum"}))
    sample = PiiDataEntry.example()
    x = gliner(sample, {"m"})
    print(sample)
    print(x)
