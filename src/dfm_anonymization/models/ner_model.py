from typing import Any, List, Optional, Set, Tuple
from bidict import bidict
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel, TokenClassifier, Tokenizer
from dfm_anonymization.util.util import filter_bio, get_torch_device


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
    def __init__(self, model: AutoModelForTokenClassification, out_bias: float = 0.0):
        self.model = model
        self.id2label = bidict(model.config.id2label)
        self.o_id = self.id2label.inv["O"]
        self.out_bias = out_bias
        self.device = torch.device(get_torch_device())
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, tokens: Any) -> List[str]:
        with torch.no_grad():
            token_batch = tokens.to(self.device)
            outputs = self.model(**token_batch)
            biased_logits = outputs.logits.clone()
            biased_logits[0, :, self.o_id] += self.out_bias
            predictions = torch.argmax(biased_logits, dim=-1)[0].tolist()
        labels = [self.id2label[prediction] for prediction in predictions]
        return labels


class NerModel(PiiModel):
    def __init__(
        self,
        name: str,
        tokenizer: Tokenizer,
        model: TokenClassifier,
    ):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.model = model

    @staticmethod
    def _load_tokenizer(tag: str):
        """Load tokenizer, working around models that specify an invalid tokenizer_class
        in their tokenizer_config.json (e.g. 'TokenizersBackend')."""
        try:
            return AutoTokenizer.from_pretrained(
                tag, use_fast=True, trust_remote_code=True
            )
        except ValueError:
            # Some models set tokenizer_class to a non-existent class (e.g. 'TokenizersBackend').
            # Load the tokenizer config, clear the bad field, and retry.
            from transformers import PreTrainedTokenizerFast

            return PreTrainedTokenizerFast.from_pretrained(tag, trust_remote_code=True)

    @staticmethod
    def from_hf(tag: str, out_bias: float = 0.0):
        bias_suffix = "" if out_bias == 0.0 else f"_biased_{out_bias}"
        name = tag.split("/")[-1] + bias_suffix
        return NerModel(
            name=name,
            tokenizer=TransformersTokenizer(tokenizer=NerModel._load_tokenizer(tag)),
            model=TransformersTokenClassifier(
                model=AutoModelForTokenClassification.from_pretrained(
                    tag, trust_remote_code=True
                ),
                out_bias=out_bias,
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
