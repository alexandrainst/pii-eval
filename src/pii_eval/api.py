from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel

from pii_eval.config import PiiEvalConfig
from pii_eval.data import PiiModel
from pii_eval.models.ensemble import Ensemble
from pii_eval.redact import redact as redact_

# Model loaded once at startup, based on the config file / env var (see
# pii_eval.config.load_config). Requests can override this on a per-call basis
# by supplying a "config" payload with the same shape as pii_eval.yaml.
default_model: PiiModel = Ensemble.default()


class RedactRequest(BaseModel):
    text: str
    config: PiiEvalConfig | None = None


class IdentifyRequest(BaseModel):
    text: str
    config: PiiEvalConfig | None = None


class RedactResponse(BaseModel):
    text: str


class SpanResponse(BaseModel):
    start: int
    end: int
    text: str
    label: str


def _resolve_model(config: PiiEvalConfig | None) -> PiiModel:
    if config is None:
        return default_model
    return Ensemble.from_config(config)


app = FastAPI(
    title="pii-eval API",
    description="Redact and identify PII (personal identifiable information) in text.",
)


@app.post("/redact", response_model=RedactResponse)
def redact_endpoint(request: RedactRequest) -> RedactResponse:
    model = _resolve_model(request.config)
    return RedactResponse(text=redact_(text=request.text, model=model))


@app.post("/identify", response_model=list[SpanResponse])
def identify_endpoint(request: IdentifyRequest) -> list[SpanResponse]:
    model = _resolve_model(request.config)
    spans = model.predict_spans(text=request.text)
    return [SpanResponse(**asdict(span)) for span in spans]
