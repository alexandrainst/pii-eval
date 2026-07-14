from dataclasses import dataclass
import os
from pathlib import Path

import requests
import yaml


CONFIG_FILE_NAME = "pii_eval_config.yaml"


@dataclass
class ClassicalConfig:
    labels: list[str] | None = None


@dataclass
class GenerativeConfig:
    active: bool = False
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    min_confidence: float = 0.0
    label_descriptions: dict[str, str] | None = None

    def is_ready(self) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 1,
                },
                timeout=1,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"warning: {e}")
            return False
        return True


@dataclass
class PiiEvalConfig:
    classical: ClassicalConfig
    generative: GenerativeConfig


def _default_config() -> PiiEvalConfig:
    return PiiEvalConfig(classical=ClassicalConfig(), generative=GenerativeConfig())


def _repo_root_config_path() -> Path:
    # src/pii_eval/config.py -> repository root
    return Path(__file__).resolve().parents[2] / CONFIG_FILE_NAME


def _resolve_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)

    env_path = os.getenv("PII_EVAL_CONFIG")
    if env_path:
        return Path(env_path)

    cwd_path = Path.cwd() / CONFIG_FILE_NAME
    if cwd_path.exists():
        return cwd_path

    return _repo_root_config_path()


def load_config(path: str | Path | None = None) -> PiiEvalConfig:
    config_path = _resolve_config_path(path)
    if not config_path.exists():
        return _default_config()

    with config_path.open("r", encoding="utf8") as handle:
        payload = yaml.safe_load(handle) or {}

    raw_classical = payload.get("classical") or {}
    classical = ClassicalConfig(
        labels=raw_classical.get("labels"),
    )

    raw_generative = payload.get("generative") or {}
    generative = GenerativeConfig(
        active=bool(raw_generative.get("active", False)),
        model=raw_generative.get("model"),
        base_url=raw_generative.get("base_url"),
        api_key=raw_generative.get("api_key"),
        min_confidence=float(raw_generative.get("min_confidence", 0.0)),
        label_descriptions=raw_generative.get("label_descriptions"),
    )

    return PiiEvalConfig(classical=classical, generative=generative)
