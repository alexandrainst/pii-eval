from pathlib import Path

from pii_eval.config import load_config


def test_load_config_defaults_when_missing(tmp_path: Path, monkeypatch) -> None:
    missing_path = tmp_path / "missing_config.yaml"
    monkeypatch.setenv("PII_EVAL_CONFIG", str(missing_path))

    config = load_config()

    assert config.generative.active is False
    assert config.generative.model is None
    assert config.generative.base_url is None
    assert config.generative.api_key is None
    assert config.generative.min_confidence == 0.0


def test_load_config_parses_generative_block(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "pii_eval_config.yaml"
    config_path.write_text(
        """
        generative:
          active: true
          model: provider/model-x
          base_url: https://api.example.com/v1/
          api_key: key-123
          min_confidence: 0.25
        """,
        encoding="utf8",
    )
    monkeypatch.setenv("PII_EVAL_CONFIG", str(config_path))

    config = load_config()

    assert config.generative.active is True
    assert config.generative.model == "provider/model-x"
    assert config.generative.base_url == "https://api.example.com/v1/"
    assert config.generative.api_key == "key-123"
    assert config.generative.min_confidence == 0.25
    assert config.generative.is_ready() is True
