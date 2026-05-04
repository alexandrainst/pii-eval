"""Tests for TransformersTokenizer, focusing on long-input chunking."""

from typing import List, Optional
from unittest.mock import MagicMock

import torch
import pytest

from dfm_anonymization.models.ner_model import TransformersTokenizer


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
