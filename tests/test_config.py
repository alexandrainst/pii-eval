from pathlib import Path

from pii_eval.config import GenerativeConfig, load_config


def test_load_config_defaults_when_missing(tmp_path: Path, monkeypatch) -> None:
    missing_path = tmp_path / "missing_config.yaml"
    monkeypatch.setenv("PII_EVAL_CONFIG", str(missing_path))

    config = load_config()

    assert config.generative is None


def test_load_config_omits_generative_block(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "pii_eval.yaml"
    config_path.write_text("other_setting: 42\n", encoding="utf8")
    monkeypatch.setenv("PII_EVAL_CONFIG", str(config_path))

    config = load_config()

    assert config.generative is None


def test_load_config_parses_generative_block(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "pii_eval.yaml"
    config_path.write_text(
        """
        generative:
          model: provider/model-x
          base_url: https://api.example.com/v1/
          api_key: key-123
          min_confidence: 0.25
        """,
        encoding="utf8",
    )
    monkeypatch.setenv("PII_EVAL_CONFIG", str(config_path))

    config = load_config()

    assert config.generative is not None
    assert config.generative.model == "provider/model-x"
    assert config.generative.base_url == "https://api.example.com/v1/"
    assert config.generative.api_key == "key-123"
    assert config.generative.min_confidence == 0.25


def test_is_ready_false_when_model_or_base_url_missing() -> None:
    assert GenerativeConfig(model=None, base_url="https://api.example.com/v1/").is_ready() is False
    assert GenerativeConfig(model="provider/model-x", base_url=None).is_ready() is False


def test_is_ready_true_when_probe_request_succeeds(monkeypatch) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            return object()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    config = GenerativeConfig(model="provider/model-x", base_url="https://api.example.com/v1/")

    assert config.is_ready() is True


def test_is_ready_false_when_probe_request_raises(monkeypatch) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("connection refused")

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    config = GenerativeConfig(model="provider/model-x", base_url="https://api.example.com/v1/")

    assert config.is_ready() is False
