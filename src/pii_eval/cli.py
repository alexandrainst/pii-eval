from dataclasses import asdict
import json

import typer

from pii_eval.config import CONFIG_FILE_NAME, load_config
from pii_eval.data import PiiModel
from pii_eval.evaluate import evaluate_all
from pii_eval.inference import generate_predictions
from pii_eval.models.ensemble import Ensemble
from pii_eval.redact import redact as redact_

app = typer.Typer()

config = load_config()
model: PiiModel = Ensemble.from_config(config)
if config.generative is not None and len(model.models) == 1:
    print(
        f"generative config present in {CONFIG_FILE_NAME} but it is not ready "
        "(generative.model/generative.base_url may be missing, or the API "
        "probe request failed). Falling back to regex based recognition."
    )


@app.command()
def run_evaluation(
    max_entries_per_config: int | None = typer.Option(
        None, "-m", "--max-entries-per-config"
    ),
):
    generate_predictions(max_entries_per_config=max_entries_per_config)
    evaluate_all(maximum_entries=max_entries_per_config)


@app.command()
def redact(text: str):
    print(redact_(text=text, model=model))


@app.command()
def identify(text: str):
    spans = model.predict_spans(text=text)
    print(json.dumps([asdict(s) for s in spans], indent=2, ensure_ascii=False))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
    port: int = typer.Option(8000, "--port", help="Port to bind to."),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload (for development)."
    ),
):
    """
    Start an API server exposing /redact and /identify endpoints.

    Both endpoints accept an optional "config" field (with the same shape as
    pii_eval.yaml) to override the generative-model settings on a per-request
    basis.
    """
    import uvicorn

    uvicorn.run("pii_eval.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
