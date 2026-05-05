from typing import Any, List, Optional, Set, Tuple
from bidict import bidict
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BatchEncoding

from dfm_anonymization.constants import MODEL_LABEL_MAP, get_model_label_map
from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel, TokenClassifier, Tokenizer
from dfm_anonymization.util.util import filter_bio, get_torch_device


class TransformersTokenizer(Tokenizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, words: List[str]) -> Tuple[Any, List[Optional[int]]]:
        max_length = self.tokenizer.model_max_length
        # Reserve space for special tokens ([CLS] and [SEP]) added per chunk.
        # We use a conservative budget; if model_max_length is unreasonably large
        # (some tokenizers set it to sys.maxsize), fall back to 512.
        if max_length > 1_000_000:
            max_length = 512
        max_tokens_per_chunk = max_length - 2  # subtract CLS + SEP

        # Split words into chunks such that each chunk fits within the model's
        # context window.  We tokenize words one-by-one to count subtokens and
        # greedily fill each chunk.
        chunks: List[List[str]] = []
        current_chunk: List[str] = []
        current_token_count = 0
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            n = len(word_tokens) if word_tokens else 1
            if current_chunk and current_token_count + n > max_tokens_per_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_token_count = 0
            current_chunk.append(word)
            current_token_count += n
        if current_chunk:
            chunks.append(current_chunk)

        if len(chunks) == 1:
            # Fast path: no chunking needed.
            encoding = self.tokenizer(
                words, is_split_into_words=True, return_tensors="pt", truncation=True
            )
            word_ids = encoding.word_ids()
            return encoding, word_ids

        # Multi-chunk path: encode each chunk, then concatenate the tensors and
        # word_id lists (removing per-chunk special tokens at the boundaries).
        all_input_ids: List = []
        all_attention_mask: List = []
        all_token_type_ids: Optional[List] = []
        all_word_ids: List[Optional[int]] = []
        word_offset = 0
        first_enc = None
        last_enc = None

        for chunk in chunks:
            enc = self.tokenizer(
                chunk, is_split_into_words=True, return_tensors="pt", truncation=True
            )
            if first_enc is None:
                first_enc = enc
            last_enc = enc

            ids = enc["input_ids"][0]  # shape (seq_len,)
            mask = enc["attention_mask"][0]  # shape (seq_len,)
            wids = enc.word_ids()  # list of Optional[int], len == seq_len

            # Strip leading/trailing special tokens (word_id is None).
            start = next(i for i, w in enumerate(wids) if w is not None)
            end = len(wids) - next(
                i for i, w in enumerate(reversed(wids)) if w is not None
            )

            inner_ids = ids[start:end]
            inner_mask = mask[start:end]
            inner_wids = [
                (w + word_offset if w is not None else None) for w in wids[start:end]
            ]

            all_input_ids.append(inner_ids)
            all_attention_mask.append(inner_mask)
            all_word_ids.extend(inner_wids)

            has_token_type = "token_type_ids" in enc
            if has_token_type:
                tti = enc["token_type_ids"][0][start:end]
                all_token_type_ids.append(tti)  # type: ignore[union-attr]
            else:
                all_token_type_ids = None  # type: ignore[assignment]

            word_offset += len(chunk)

        # Re-add a single CLS at the start and SEP at the end using the first/last
        # chunk's encodings so that the model sees the expected special tokens.
        cls_id = first_enc["input_ids"][0, :1]
        sep_id = last_enc["input_ids"][0, -1:]
        cls_mask = first_enc["attention_mask"][0, :1]
        sep_mask = last_enc["attention_mask"][0, -1:]

        cat_ids = torch.cat([cls_id] + all_input_ids + [sep_id], dim=0).unsqueeze(0)
        cat_mask = torch.cat(
            [cls_mask] + all_attention_mask + [sep_mask], dim=0
        ).unsqueeze(0)
        final_word_ids: List[Optional[int]] = [None] + all_word_ids + [None]

        result = {"input_ids": cat_ids, "attention_mask": cat_mask}

        if all_token_type_ids is not None:
            cls_tti = first_enc["token_type_ids"][0, :1]
            sep_tti = last_enc["token_type_ids"][0, -1:]
            cat_tti = torch.cat(
                [cls_tti] + all_token_type_ids + [sep_tti], dim=0
            ).unsqueeze(0)
            result["token_type_ids"] = cat_tti

        # Build a BatchEncoding-like object that also supports .word_ids().
        # We need a BatchEncoding that returns our patched word_ids.
        # The simplest way is to construct one and monkey-patch word_ids().
        batch_enc = BatchEncoding(result)
        _final_word_ids = final_word_ids  # capture for closure

        def _word_ids(batch_index: int = 0) -> List[Optional[int]]:  # noqa: ARG001
            return _final_word_ids

        batch_enc.word_ids = _word_ids  # type: ignore[method-assign]

        return batch_enc, final_word_ids


class TransformersTokenClassifier(TokenClassifier):
    def __init__(
        self,
        model: AutoModelForTokenClassification,
        o_bias: float = 0.0,
        label_subset: set[str] | None = None,
    ):
        self.model = model
        self.id2label = bidict(model.config.id2label)
        self.o_id = self.id2label.inv["O"]
        self.o_bias = o_bias
        self.device = torch.device(get_torch_device())
        self.model.to(self.device)
        self.model.eval()

        self.label_subset = label_subset
        num_labels = len(self.id2label)
        if label_subset is not None:
            mask = torch.full((num_labels,), float("-inf"))
            for label_id, label in self.id2label.items():
                if label == "O" or label[2:] in label_subset:
                    mask[label_id] = 0.0
        else:
            mask = torch.zeros(num_labels)
        self.logit_mask = mask.to(self.device)

    def __call__(self, tokens: Any) -> List[str]:
        # Determine the true maximum sequence length the model can handle.
        # RoBERTa-family models (including XLM-RoBERTa) reserve position IDs
        # 0 and 1 (padding_idx=1), so actual tokens use positions 2..N, meaning
        # only max_position_embeddings - 2 tokens can be processed at once.
        # We read padding_idx from the embeddings to handle this correctly
        # instead of hard-coding model families.
        try:
            pos_emb = self.model.base_model.embeddings.position_embeddings
            padding_idx = pos_emb.padding_idx
            if not isinstance(padding_idx, int):
                padding_idx = 0
            position_offset = padding_idx + 1  # first usable position id
        except AttributeError:
            position_offset = 0
        max_length = (
            getattr(self.model.config, "max_position_embeddings", 512) - position_offset
        )

        token_batch = tokens.to(self.device)
        seq_len = token_batch["input_ids"].shape[1]

        if seq_len <= max_length:
            # Fast path: sequence fits in the model's context window.
            with torch.no_grad():
                outputs = self.model(**token_batch)
            all_logits = outputs.logits  # (1, seq_len, num_labels)
        else:
            # Slow path: the concatenated encoding produced by
            # TransformersTokenizer is a single flat sequence
            # [CLS token1 token2 ... tokenN SEP].  Split it into
            # non-overlapping chunks of max_length and concatenate the
            # resulting logits — no tokens are shared between chunks so
            # no stitching overlap is needed.
            chunk_logits_list: List[torch.Tensor] = []
            start = 0
            while start < seq_len:
                end = min(start + max_length, seq_len)
                chunk = {k: v[:, start:end] for k, v in token_batch.items()}
                with torch.no_grad():
                    chunk_out = self.model(**chunk)
                chunk_logits_list.append(chunk_out.logits)
                start = end

            all_logits = torch.cat(chunk_logits_list, dim=1)  # (1, seq_len, num_labels)

        # this part remains the same
        biased_logits = all_logits.clone()
        biased_logits[0, :, self.o_id] += self.o_bias
        biased_logits += self.logit_mask
        predictions = torch.argmax(biased_logits, dim=-1)[0].tolist()
        labels = [self.id2label[prediction] for prediction in predictions]
        return labels


class NerModel(PiiModel):
    def __init__(
        self,
        name: str,
        tokenizer: Tokenizer,
        model: TokenClassifier,
        label_subset: set[str] | None = None,
    ):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.model = model
        self.label_subset = label_subset

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
        map = get_model_label_map(name)
        return NerModel(
            name=name,
            tokenizer=TransformersTokenizer(tokenizer=NerModel._load_tokenizer(tag)),
            model=TransformersTokenClassifier(
                model=AutoModelForTokenClassification.from_pretrained(
                    tag, trust_remote_code=True
                ),
                o_bias=out_bias,
                label_subset=set(map.keys()) if map else None,
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
        if len(bio_tags) > 500:
            pass
        return filter_bio(bio_tags, label_whitelist) if label_whitelist else bio_tags
