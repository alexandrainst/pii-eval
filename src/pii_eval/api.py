from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from pii_eval.config import PiiEvalConfig, load_config
from pii_eval.data import PiiModel
from pii_eval.models.ensemble import Ensemble
from pii_eval.redact import redact as redact_
from pii_eval.span import LabelSpan

# Config + model loaded once at startup, based on the config file / env var (see
# pii_eval.config.load_config). Requests can override this on a per-call basis
# by supplying a "config" payload with the same shape as pii_eval.yaml.
server_config: PiiEvalConfig = load_config()
default_model: PiiModel = Ensemble.from_config(server_config)


class PiiRequest(BaseModel):
    text: str = Field(
        examples=["My name is Jane Doe and my email is jane.doe@example.com."]
    )
    config: PiiEvalConfig | None = Field(default=None, examples=[None])


class RedactResponse(BaseModel):
    text: str


class ConfigResponse(BaseModel):
    """The server's active config, with the generative API key stripped out so
    it never reaches the browser. `api_key_configured` lets the client know a
    key is available server-side (it can then send a blank key, which the server
    fills in from `server_config`)."""

    config: PiiEvalConfig
    api_key_configured: bool


def _fill_secrets(config: PiiEvalConfig) -> PiiEvalConfig:
    """Fill a blank generative API key from the server-side config, so the real
    key never has to travel to the client and back."""
    gen = config.generative
    server_gen = server_config.generative
    if gen is not None and not gen.api_key and server_gen is not None:
        config = config.model_copy(
            update={"generative": gen.model_copy(update={"api_key": server_gen.api_key})}
        )
    return config


def _resolve_model(config: PiiEvalConfig | None) -> PiiModel:
    if config is None:
        return default_model
    return Ensemble.from_config(_fill_secrets(config))


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


@app.get("/config", response_model=ConfigResponse)
def config_endpoint() -> ConfigResponse:
    """Return the server's active config so a client (e.g. the demo UI) can
    pre-populate itself from pii_eval.yaml. The generative API key is redacted."""
    safe = server_config.model_copy(deep=True)
    if safe.generative is not None:
        safe.generative = safe.generative.model_copy(update={"api_key": ""})
    return ConfigResponse(
        config=safe,
        api_key_configured=bool(
            server_config.generative is not None and server_config.generative.api_key
        ),
    )


class _NoCacheStaticFiles(StaticFiles):
    """Serve the demo assets with caching disabled, so a running server always
    hands out the current HTML/JS/CSS instead of a stale copy the browser cached
    from an earlier session."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


# Web UI demo (src/demo). Served on the same origin as the API so the page can
# call /identify without CORS. Mounted only if the directory is present (it is
# not shipped inside the installed package).
_DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
if _DEMO_DIR.is_dir():
    app.mount("/demo", _NoCacheStaticFiles(directory=_DEMO_DIR, html=True), name="demo")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/demo/")
