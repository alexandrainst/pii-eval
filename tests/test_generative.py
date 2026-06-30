import re

from pii_eval.models.generative import NEREntityFormat, NERResponse
from pii_eval.span import TextSpan


def _whitespace_tokenizer(text: str) -> list[TextSpan]:
    return [
        TextSpan(text=match.group(0), start=match.start(), end=match.end())
        for match in re.finditer(r"\S+", text)
    ]


def test_to_bio_matches_repeated_entities_in_output_order() -> None:
    text = "Alice met Alice. His name is John Johnson."
    response = NERResponse(
        entities=[
            NEREntityFormat(text="Alice", label="PERSON_NAME", confidence=0.95),
            NEREntityFormat(text="Alice", label="PERSON_NAME", confidence=0.90),
            NEREntityFormat(text="John", label="PERSON_NAME", confidence=0.90),
            NEREntityFormat(text="Johnson", label="PERSON_NAME", confidence=0.90),
        ]
    )

    bio = response.to_bio(text, tokenizer=_whitespace_tokenizer)

    assert [(token.text, token.label) for token in bio] == [
        ("Alice", "B-PERSON_NAME"),
        ("met", "O"),
        ("Alice.", "B-PERSON_NAME"),
        ("His", "O"),
        ("name", "O"),
        ("is", "O"),
        ("John", "B-PERSON_NAME"),
        ("Johnson.", "I-PERSON_NAME"),
    ]


def test_to_bio_applies_min_confidence_filter() -> None:
    text = "Call 12345 now"
    response = NERResponse(
        entities=[
            NEREntityFormat(text="12345", label="PHONE_NUMBER", confidence=0.20),
        ]
    )

    bio = response.to_bio(text, tokenizer=_whitespace_tokenizer, min_confidence=0.30)

    assert [(token.text, token.label) for token in bio] == [
        ("Call", "O"),
        ("12345", "O"),
        ("now", "O"),
    ]
