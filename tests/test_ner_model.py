"""Tests for TransformersTokenizer and TransformersTokenClassifier, focusing on long-input chunking."""

from typing import List, Optional
from unittest.mock import MagicMock

import torch
import pytest

from dfm_anonymization.models.ner_model import (
    TransformersTokenClassifier,
    TransformersTokenizer,
)


def _make_mock_tokenizer(vocab_size: int = 100, model_max_length: int = 10):
    """Return a minimal mock of a HuggingFace fast tokenizer.

    Each word tokenizes to exactly one subtoken with id = ord(word[0]) % vocab_size.
    Special tokens: CLS = 0, SEP = 1.
    """
    from transformers import BatchEncoding

    mock = MagicMock()
    mock.model_max_length = model_max_length

    def tokenize(word):
        # Each word → single subtoken (simple mock)
        return [f"▁{word}"]

    mock.tokenize.side_effect = tokenize

    def encode_words(
        words, is_split_into_words=True, return_tensors="pt", truncation=True
    ):
        # Build a simple encoding: [CLS] + one token per word + [SEP]
        cls_id, sep_id = 0, 1
        word_token_ids = [ord(w[0]) % 100 + 2 for w in words]
        all_ids = [cls_id] + word_token_ids + [sep_id]
        ids_tensor = torch.tensor([all_ids])
        mask_tensor = torch.ones_like(ids_tensor)

        # word_ids: None for CLS/SEP, index for real tokens
        raw_word_ids = [None] + list(range(len(words))) + [None]

        enc = BatchEncoding({"input_ids": ids_tensor, "attention_mask": mask_tensor})
        enc.word_ids = lambda batch_index=0: raw_word_ids  # type: ignore[method-assign]
        return enc

    mock.side_effect = encode_words
    return mock


class TestTransformersTokenizerChunking:
    def test_short_input_returns_all_word_ids(self):
        """Words that fit in one chunk: word_ids covers all words."""
        inner = _make_mock_tokenizer(model_max_length=20)
        tok = TransformersTokenizer(inner)
        words = ["hello", "world"]
        _, word_ids = tok(words)
        non_none = [w for w in word_ids if w is not None]
        assert set(non_none) == {0, 1}

    def test_long_input_all_words_covered(self):
        """When input exceeds model_max_length, every word must appear in word_ids."""
        # model_max_length=6 → max_tokens_per_chunk=4 → chunks of 4 words each
        inner = _make_mock_tokenizer(model_max_length=6)
        tok = TransformersTokenizer(inner)
        # 10 words → requires at least 3 chunks
        words = [f"w{i}" for i in range(10)]
        _, word_ids = tok(words)
        non_none = [w for w in word_ids if w is not None]
        # Every word index 0..9 must appear exactly once
        assert sorted(non_none) == list(range(len(words)))

    def test_long_input_word_ids_monotonically_increasing(self):
        """word_ids (ignoring Nones) should be strictly increasing 0..N-1."""
        inner = _make_mock_tokenizer(model_max_length=6)
        tok = TransformersTokenizer(inner)
        words = [f"w{i}" for i in range(12)]
        _, word_ids = tok(words)
        non_none = [w for w in word_ids if w is not None]
        assert non_none == list(range(len(words)))

    def test_encoding_input_ids_length_matches_word_ids(self):
        """The input_ids tensor length must equal len(word_ids)."""
        inner = _make_mock_tokenizer(model_max_length=6)
        tok = TransformersTokenizer(inner)
        words = [f"w{i}" for i in range(10)]
        encoding, word_ids = tok(words)
        assert encoding["input_ids"].shape[1] == len(word_ids)


def _make_mock_classifier(num_labels: int = 4, max_position_embeddings: int = 10):
    """Return a minimal mock of a HuggingFace token classification model.

    The model always predicts label 0 for every token.  max_position_embeddings
    controls the context window enforced by TransformersTokenClassifier.
    """
    id2label = {i: (f"B-ENT{i}" if i > 0 else "O") for i in range(num_labels)}

    config = MagicMock()
    config.id2label = id2label
    config.max_position_embeddings = max_position_embeddings

    model = MagicMock()
    model.config = config

    def forward(**inputs):
        seq_len = inputs["input_ids"].shape[1]
        device = inputs["input_ids"].device
        logits = torch.zeros(1, seq_len, num_labels, device=device)
        out = MagicMock()
        out.logits = logits
        return out

    model.side_effect = forward
    # .to() and .eval() must be no-ops that return the mock itself
    model.to.return_value = model
    model.eval.return_value = model

    return model


class TestTransformersTokenClassifierChunking:
    def test_short_input_returns_label_per_token(self):
        """Sequences within max_position_embeddings get one label per token."""
        model = _make_mock_classifier(max_position_embeddings=20)
        classifier = TransformersTokenClassifier(model)
        device = classifier.device
        seq_len = 8
        input_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        token_dict = {"input_ids": input_ids}

        tokens = MagicMock()
        tokens.to.return_value = token_dict

        labels = classifier(tokens)
        assert len(labels) == seq_len

    def test_long_input_returns_label_per_token(self):
        """Sequences exceeding max_position_embeddings still get one label per token."""
        max_pos = 6
        model = _make_mock_classifier(max_position_embeddings=max_pos)
        classifier = TransformersTokenClassifier(model)
        device = classifier.device
        seq_len = 20  # much longer than max_pos
        input_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        token_dict = {"input_ids": input_ids}

        tokens = MagicMock()
        tokens.to.return_value = token_dict

        labels = classifier(tokens)
        assert len(labels) == seq_len
