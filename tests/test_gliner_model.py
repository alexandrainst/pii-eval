from bidict import bidict

from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.models.gliner_model import GlinerModel


class _FakeGlinerBackend:
    def __init__(self) -> None:
        self.calls = 0

    def batch_predict_entities(self, texts: list[str], labels: list[str]) -> list[list[dict]]:
        self.calls += 1
        assert texts == ["Alice went home"]
        assert labels == ["navn"]
        return [[{"start": 0, "end": 5, "label": "navn"}]]


def test_gliner_model_loads_once_and_maps_spans(monkeypatch) -> None:
    backend = _FakeGlinerBackend()
    load_calls = {"count": 0}

    def _load_model(self) -> _FakeGlinerBackend:
        load_calls["count"] += 1
        return backend

    monkeypatch.setattr(GlinerModel, "_load_model", _load_model)

    model = GlinerModel(tag="nvidia/gliner-pii", labels=bidict({"PERSON_NAME": "navn"}))
    entry = PiiDataEntry(text="Alice went home", tokens=["Alice", "went", "home"])

    first = model(entry, {"PERSON_NAME"})
    second = model(entry)

    assert first == ["B-PERSON_NAME", "O", "O"]
    assert second == ["B-PERSON_NAME", "O", "O"]
    assert load_calls["count"] == 1
    assert backend.calls == 2