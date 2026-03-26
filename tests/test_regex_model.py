import re

from dfm_anonymization.constants import (
    ISO_LABEL_BANK_ACCOUNT,
    ISO_LABEL_DRIVERS_LICENSE,
    ISO_LABEL_EMAIL_ADDRESS,
    ISO_LABEL_IP_ADDRESS,
    ISO_LABEL_PASSPORT,
    ISO_LABEL_PCI,
    ISO_LABEL_PHONE_NUMBER,
    ISO_LABEL_PRECISE_LOCATION,
    ISO_LABEL_SSN,
    ISO_LABEL_VEHICLE_ID,
)
from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.models.regex_model import RegexModel


def _make_entry(text: str) -> PiiDataEntry:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return PiiDataEntry(text=text, tokens=tokens)


def _extract_entities(entry: PiiDataEntry, bio_tags: list[str]) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    token_spans = entry.get_token_spans()
    active_label: str | None = None
    start: int | None = None
    end: int | None = None

    for tag, (tok_start, tok_end) in zip(bio_tags, token_spans):
        if tag == "O":
            if active_label is not None and start is not None and end is not None:
                entities.append((active_label, entry.text[start:end]))
            active_label = None
            start = None
            end = None
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B" or label != active_label:
            if active_label is not None and start is not None and end is not None:
                entities.append((active_label, entry.text[start:end]))
            active_label = label
            start = tok_start
        end = tok_end

    if active_label is not None and start is not None and end is not None:
        entities.append((active_label, entry.text[start:end]))

    return entities


def test_regex_model_detects_structured_pii_in_example_entry() -> None:
    entry = PiiDataEntry.example_pii()
    model = RegexModel()

    predicted_tags = model(entry)
    entities = _extract_entities(entry, predicted_tags)

    assert (ISO_LABEL_IP_ADDRESS, "10.0.0.42") in entities
    assert (ISO_LABEL_EMAIL_ADDRESS, "frederik.sorensen@firma.dk") in entities
    assert any(
        label == ISO_LABEL_PRECISE_LOCATION and "Blegdamsvej 24, 2100 København Ø" in text
        for label, text in entities
    )


def test_regex_model_detects_mixed_english_and_danish_identifiers() -> None:
    entry = _make_entry(
        "Call +45 12 34 56 78 or +1 (202) 555-0199. CPR 120598-1234, SSN 123-45-6789, "
        "IBAN DK5000400440116243, card 4111 1111 1111 1111, passport number 123456789, "
        "driver's license DL-042-778-991, VIN 1HGCM82633A004352, coordinates 55.6761, 12.5683."
    )
    model = RegexModel()

    predicted_tags = model(entry)
    entities = _extract_entities(entry, predicted_tags)
    labels = {label for label, _ in entities}

    assert ISO_LABEL_PHONE_NUMBER in labels
    assert ISO_LABEL_SSN in labels
    assert ISO_LABEL_BANK_ACCOUNT in labels
    assert ISO_LABEL_PCI in labels
    assert ISO_LABEL_PASSPORT in labels
    assert ISO_LABEL_DRIVERS_LICENSE in labels
    assert ISO_LABEL_VEHICLE_ID in labels
    assert ISO_LABEL_PRECISE_LOCATION in labels


def test_regex_model_respects_label_whitelist() -> None:
    entry = _make_entry(
        "Email jane.doe@example.com or call +45 12 34 56 78. Coordinates 55.6761, 12.5683."
    )
    model = RegexModel()

    predicted_tags = model(entry, label_whitelist={ISO_LABEL_PHONE_NUMBER})
    entities = _extract_entities(entry, predicted_tags)

    assert entities == [(ISO_LABEL_PHONE_NUMBER, "+45 12 34 56 78")]
