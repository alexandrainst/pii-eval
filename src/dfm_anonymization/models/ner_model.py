from typing import Any, List, Optional, Set, Tuple
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel, TokenClassifier, Tokenizer
from dfm_anonymization.util import filter_bio, get_torch_device


class TransformersTokenizer(Tokenizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, words: List[str]) -> Tuple[Any, List[Optional[int]]]:
        encoding = self.tokenizer(
            words, is_split_into_words=True, return_tensors="pt", truncation=True
        )
        word_ids = encoding.word_ids()
        return encoding, word_ids


class TransformersTokenClassifier(TokenClassifier):
    def __init__(self, model: AutoModelForTokenClassification):
        self.model = model
        self.id2label = model.config.id2label
        self.device = torch.device(get_torch_device())
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, tokens: Any) -> List[str]:
        with torch.no_grad():
            token_batch = tokens.to(self.device)
            outputs = self.model(**token_batch)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
        return [self.id2label[prediction] for prediction in predictions]


class NerModel(PiiModel):
    def __init__(self, name: str, tokenizer: Tokenizer, model: TokenClassifier):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.model = model

    @staticmethod
    def from_hf(tag: str):
        return NerModel(
            name=tag.split("/")[-1],
            tokenizer=TransformersTokenizer(
                AutoTokenizer.from_pretrained(tag, use_fast=True)
            ),
            model=TransformersTokenClassifier(
                AutoModelForTokenClassification.from_pretrained(tag)
            ),
        )

    def __call__(
        self, entry: PiiDataEntry, label_whitelist: Set[str] | None = None
    ) -> List[str]:
        words = entry.tokens
        subtokens, word_ids = self.tokenizer(words)
        sublabels = self.model(subtokens)
        bio_tags = ["O"] * len(words)
        current_word_id = -1
        for sublabel, word_id in zip(sublabels, word_ids):
            if word_id is None or word_id == current_word_id:
                continue
            if word_id >= len(bio_tags):
                continue
            current_word_id = word_id
            if sublabel.startswith("B-"):
                bio_tags[word_id] = sublabel
            elif sublabel.startswith("I-"):
                entity = sublabel[2:]
                prev_label = bio_tags[word_id - 1] if word_id > 0 else None
                if prev_label and prev_label.endswith(entity):
                    bio_tags[word_id] = sublabel
                else:
                    bio_tags[word_id] = "B-" + entity
            else:
                bio_tags[word_id] = "O"
        return filter_bio(bio_tags, label_whitelist) if label_whitelist else bio_tags
