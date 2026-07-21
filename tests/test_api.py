from fastapi.testclient import TestClient

import pii_eval.api as api
from pii_eval.models.regex_model import Regex

client = TestClient(api.app)

# Explicitly omit the generative block in these requests so tests stay
# deterministic and offline (the repo's local pii_eval.yaml, if present, may
# have it turned on).
REGEX_ONLY_CONFIG = {}


def test_redact_endpoint_masks_pii() -> None:
    response = client.post(
        "/redact",
        json={
            "text": "Navnet er Anders Sand. Og 21324354 er telefonnummeret.",
            "config": REGEX_ONLY_CONFIG,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "<PHONE_NUMBER>" in body["text"]


def test_identify_endpoint_returns_spans() -> None:
    response = client.post(
        "/identify",
        json={
            "text": "Ring på 21324354.",
            "config": REGEX_ONLY_CONFIG,
        },
    )

    assert response.status_code == 200
    spans = response.json()
    assert any(span["label"] == "PHONE_NUMBER" for span in spans)
    for span in spans:
        assert set(span.keys()) == {"start", "end", "text", "label"}


def test_identify_endpoint_without_config_uses_default_model(monkeypatch) -> None:
    # Swap out the module-level default model (loaded from whatever
    # pii_eval.yaml is on disk) for a deterministic, offline one, to confirm
    # that omitting "config" falls back to it instead of building a new one.
    monkeypatch.setattr(api, "default_model", Regex())

    response = client.post("/identify", json={"text": "Hej verden."})

    assert response.status_code == 200
    assert response.json() == []
