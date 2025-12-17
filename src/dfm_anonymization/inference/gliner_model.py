from typing import Set, Tuple
import torch
from gliner import GLiNER

from dfm_anonymization.inference.interfaces import PiiModel
from dfm_anonymization.util.util import get_torch_device


example_models = ["nvidia/gliner-pii", "urchade/gliner_multi-v2.1"]


class GlinerModel(PiiModel):
    def __init__(self, tag: str):
        self.model = GLiNER.from_pretrained(tag, map_location=get_torch_device())
        raise NotImplemented

    def __call__(self, input: Tuple[str, Set[str]]):
        text, labels = input
        x = self.model.predict_entities(text, labels)
        raise NotImplemented
        return
