from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from pii_eval.config import PiiEvalConfig
from pii_eval.data import PiiModel
from pii_eval.models.ensemble import Ensemble
from pii_eval.redact import redact as redact_
from pii_eval.span import LabelSpan

# Model loaded once at startup, based on the config file / env var (see
# pii_eval.config.load_config). Requests can override this on a per-call basis
# by supplying a "config" payload with the same shape as pii_eval.yaml.
default_model: PiiModel = Ensemble.default()


class PiiRequest(BaseModel):
    text: str = Field(
        examples=["My name is Jane Doe and my email is jane.doe@example.com."]
    )
    config: PiiEvalConfig


class RedactResponse(BaseModel):
    text: str


def _resolve_model(config: PiiEvalConfig | None) -> PiiModel:
    if config is None:
        return default_model
    return Ensemble.from_config(config)


app = FastAPI(
    title="pii-eval API",
    description="Redact and identify PII (personal identifiable information) in text.",
)


@app.post("/redact", response_model=RedactResponse)
def redact_endpoint(request: PiiRequest) -> RedactResponse:
    model = _resolve_model(request.config)
    return RedactResponse(text=redact_(text=request.text, model=model))


@app.post("/identify", response_model=list[LabelSpan])
def identify_endpoint(request: PiiRequest) -> list[LabelSpan]:
    model = _resolve_model(request.config)
    return model.predict_spans(text=request.text)
