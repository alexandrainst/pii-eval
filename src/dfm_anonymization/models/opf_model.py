"""Wrapper around the openai/privacy-filter (opf) model as a PiiModel."""

from typing import List, Set

from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.util.util import filter_bio, get_torch_device


# opf label -> BIO prefix mapping.
# opf uses BIOES tagging; we normalise E-/S- to I-/B- respectively so downstream
# code that expects plain BIO tags works unchanged.
_BIOES_TO_BIO = {
    "B": "B",
    "I": "I",
    "E": "I",  # end-of-span token → inside
    "S": "B",  # single-token span → begin
    "O": "O",
}


class OpfNerModel(PiiModel):
    """PiiModel wrapper for the openai/privacy-filter model via the opf package.

    The opf package ships its own model loading/runtime stack that is completely
    separate from transformers, so we cannot use NerModel.from_hf() for this
    checkpoint.  Instead we use the opf Python API directly and translate its
    character-level span output back to word-level BIO tags.
    """

    def __init__(self, device: str | None = None) -> None:
        super().__init__("privacy-filter")
        from opf._api import (
            OPF,
        )  # imported lazily so the class can be imported without opf

        self._opf = OPF(device=device or get_torch_device())

    def __call__(
        self, entry: PiiDataEntry, label_map: dict[str, str] | None = None
    ) -> List[str]:
        # opf works on raw text; rebuild it from word tokens with single spaces.
        text = " ".join(entry.tokens)

        result = self._opf.redact(text)
        # result is a RedactionResult; result.detected_spans is a tuple of DetectedSpan
        # with .start/.end (char offsets into result.text) and .label.

        # Build a char → word-index map so we can translate spans back to BIO tags.
        char_to_word: list[int] = []
        for word_idx, word in enumerate(entry.tokens):
            char_to_word.extend([word_idx] * len(word))
            char_to_word.append(-1)  # space between words
        # Remove trailing phantom space added after last word
        if char_to_word and char_to_word[-1] == -1:
            char_to_word.pop()

        bio_tags = ["O"] * len(entry.tokens)

        for span in result.detected_spans:
            label = span.label
            # Walk every character in the span and mark the corresponding words.
            prev_word_idx = -1
            for char_idx in range(span.start, min(span.end, len(char_to_word))):
                word_idx = char_to_word[char_idx]
                if word_idx < 0 or word_idx >= len(bio_tags):
                    continue
                if word_idx == prev_word_idx:
                    continue
                if prev_word_idx == -1:
                    bio_tags[word_idx] = f"B-{label}"
                else:
                    bio_tags[word_idx] = f"I-{label}"
                prev_word_idx = word_idx

        return filter_bio(bio_tags, label_map) if label_map else bio_tags
