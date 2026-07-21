from dataclasses import dataclass
import os
from pathlib import Path

import requests
import yaml


CONFIG_FILE_NAME = "pii_eval.yaml"


@dataclass
class ClassicalConfig:
    labels: list[str] | None = None


@dataclass
class GenerativeConfig:
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    min_confidence: float = 0.0
    label_descriptions: dict[str, str] | None = None

    def is_ready(self) -> bool:
        """Check that the generative API is actually reachable and usable.

        Probes it with a minimal (1-token) completion request rather than
        just checking that model/base_url are set, since a misconfigured or
        unreachable endpoint should be treated the same as "not ready".
        """
        if not (self.model and self.base_url):
            return False
        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=5.0)
            client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False


@dataclass
class PiiEvalConfig:
    classical: ClassicalConfig = ClassicalConfig()
    generative: GenerativeConfig | None = None


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
        return PiiEvalConfig()

    with config_path.open("r", encoding="utf8") as handle:
        payload = yaml.safe_load(handle) or {}

    raw_classical = payload.get("classical") or None
    classical = (
        ClassicalConfig()
        if raw_classical is None
        else ClassicalConfig(
            labels=raw_classical.get("labels"),
        )
    )

    raw_generative = payload.get("generative") or None
    generative = (
        None
        if raw_generative is None
        else GenerativeConfig(
            model=raw_generative.get("model"),
            base_url=raw_generative.get("base_url"),
            api_key=raw_generative.get("api_key"),
            min_confidence=float(raw_generative.get("min_confidence", 0.0)),
            label_descriptions=raw_generative.get("label_descriptions"),
        )
    )

    return PiiEvalConfig(classical=classical, generative=generative)
