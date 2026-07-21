from pii_eval.data import PiiEntry
from pii_eval.models.pn_lookup import PNLookup
from pii_eval.span import spans_to_bio


def _make_entry(text: str) -> PiiEntry:
    # Reuse project tokenization so predicted BIO aligns with evaluation assumptions.
    return PiiEntry(
        uid="test", text=text, bio=spans_to_bio(text=text, spans=[]), meta={}
    )


def _labeled_tokens(entry: PiiEntry) -> list[tuple[str, str]]:
    return [(token.text, token.label) for token in entry.bio if token.label != "O"]


def test_pn_lookup_detects_first_last_with_extra_whitespace_and_case_insensitivity() -> (
    None
):
    model = PNLookup()
    output = model(_make_entry("Kontakt jens     hansen i dag."))

    assert _labeled_tokens(output) == [
        ("jens", "B-PERSON_NAME"),
        ("    ", "I-PERSON_NAME"),
        ("hansen", "I-PERSON_NAME"),
    ]


def test_pn_lookup_detects_reversed_last_comma_first() -> None:
    model = PNLookup()
    output = model(_make_entry("Hansen, Jens deltog i mødet."))

    assert _labeled_tokens(output) == [
        ("Hansen", "B-PERSON_NAME"),
        (",", "I-PERSON_NAME"),
        ("Jens", "I-PERSON_NAME"),
    ]


def test_pn_lookup_expands_longer_name_sequences() -> None:
    model = PNLookup()
    output = model(_make_entry("Kære Fru Mirjana Run Venzo Caraballo Gaudart, tak."))

    assert _labeled_tokens(output) == [
        ("Mirjana", "B-PERSON_NAME"),
        ("Run", "I-PERSON_NAME"),
        ("Venzo", "I-PERSON_NAME"),
        ("Caraballo", "I-PERSON_NAME"),
        ("Gaudart", "I-PERSON_NAME"),
    ]


def test_pn_lookup_uses_field_prefix_context_for_single_known_name() -> None:
    model = PNLookup()
    output = model(_make_entry("Registrering: Fornavn: Yelda Efternavn: Schawohl"))

    assert ("Yelda", "B-PERSON_NAME") in _labeled_tokens(output)
    assert not any(token == "Efternavn" for token, _ in _labeled_tokens(output))


def test_pn_lookup_uses_greeting_context_for_single_known_name() -> None:
    model = PNLookup()
    output = model(_make_entry("Hej Annine! Velkommen."))

    assert _labeled_tokens(output) == [("Annine", "B-PERSON_NAME")]


def test_pn_lookup_matches_lowercase_known_surname_without_context() -> None:
    model = PNLookup()
    output = model(_make_entry("Det må virkelig have været drøjt for jacobsen,"))

    assert ("jacobsen", "B-PERSON_NAME") in _labeled_tokens(output)
