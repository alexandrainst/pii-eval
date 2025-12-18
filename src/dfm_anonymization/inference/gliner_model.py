from typing import Any, Dict, List, Optional, Set
from gliner import GLiNER
from bidict import bidict

from dfm_anonymization.inference.dataset import PiiDataEntry
from dfm_anonymization.inference.interfaces import PiiModel
from dfm_anonymization.util.util import get_torch_device


GLINER_MODEL_EXAMPLES = ["nvidia/gliner-pii", "urchade/gliner_multi-v2.1"]


def _tokens_to_offsets(text: str, tokens: List[str]) -> List[tuple[int, int]]:
    offsets: List[tuple[int, int]] = []
    cursor = 0
    text_len = len(text)
    for token in tokens:
        if not token:
            offsets.append((cursor, cursor))
            continue
        while cursor < text_len and text[cursor].isspace():
            cursor += 1
        end = cursor + len(token)
        if text[cursor:end] == token:
            start = cursor
        else:
            start = text.find(token, cursor)
            if start == -1:
                raise ValueError(f"Token '{token}' not found in text segment.")
            end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def _spans_to_bio(token_spans: List[tuple[int, int]],
                  spans: List[Dict[str, Any]]) -> List[str]:
    tags = ["O"] * len(token_spans)
    for span in sorted(spans,
                       key=lambda item: (int(item["start"]), int(item["end"]))):
        start = int(span["start"])
        end = int(span["end"])
        label = span["label"]
        inside = False
        for idx, (tok_start, tok_end) in enumerate(token_spans):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            if tok_start < end and tok_end > start:
                prefix = "B" if not inside else "I"
                if tags[idx] == "O":
                    tags[idx] = f"{prefix}-{label}"
                inside = True
    return tags


class GlinerModel(PiiModel):
    def __init__(
            self,
            tag: str,
            labels: Set[str] | bidict[str, str],
    ):
        self.name = tag.split('/')[-1]
        self.model = GLiNER.from_pretrained(
            tag, map_location=get_torch_device())
        self.labels = labels

    def __call__(self, entry: PiiDataEntry) -> List[str]:
        descriptions = list(self.labels) if isinstance(self.labels, Set) \
            else list(self.labels.values())
        spans: List[Dict[str, Any]] = \
            self.model.predict_entities(entry.text, descriptions)
        if isinstance(self.labels, bidict):
            for span in spans:
                span['label'] = self.labels.inverse[span['label']]
        bio_tags = _spans_to_bio(
            _tokens_to_offsets(entry.text, entry.tokens),
            spans
        )
        return bio_tags


if __name__ == "__main__":
    gliner = GlinerModel(GLINER_MODEL_EXAMPLES[0], bidict({"n": "name"}))
    x = gliner(PiiDataEntry.sample())
    print(x)
