import os
from pathlib import Path

from pydantic import BaseModel, Field
import yaml

from pii_eval.constants import *


CONFIG_FILE_NAME = "pii_eval.yaml"


class ClassicalConfig(BaseModel):
    labels: list[str] = Field(default=list(REGEX_FEASIBLE_LABELS))


class GenerativeConfig(BaseModel):
    model: str = Field(examples=["gpt-4o"])
    base_url: str = Field(examples=["https://api.openai.com/v1"])
    api_key: str = Field(examples=["sk-..."])
    min_confidence: float = Field(default=0.0)
    label_descriptions: dict[str, str] = Field(default=NON_REGEX_LABEL_DESCRIPTIONS_DA)

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


class PiiEvalConfig(BaseModel):
    classical: ClassicalConfig = Field(default_factory=ClassicalConfig)
    generative: GenerativeConfig | None = Field(default=None, examples=[None])


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

    # Drop empty blocks (e.g. a bare "classical:" key) so they fall back to
    # field defaults instead of failing validation as an explicit null.
    payload = {key: value for key, value in payload.items() if value is not None}

    return PiiEvalConfig.model_validate(payload)
