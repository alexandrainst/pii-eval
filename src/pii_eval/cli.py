import json

import typer

from pii_eval.config import CONFIG_FILE_NAME, load_config
from pii_eval.data import PiiModel
from pii_eval.evaluate import evaluate_all
from pii_eval.inference import generate_predictions
from pii_eval.models.ensemble import Ensemble
from pii_eval.redact import redact as redact_

app = typer.Typer()


def _get_model(config_path: str = CONFIG_FILE_NAME) -> PiiModel:
    config = load_config(config_path)
    gen_config = config.generative
    if gen_config is not None and not gen_config.is_ready():
        print(
            f"generative config found in {CONFIG_FILE_NAME} but "
            "failed to contact api. "
            "Falling back to regex based recognition."
        )
    return Ensemble.from_config(config)


@app.command()
def run_evaluation(
    override: bool = typer.Option("-o", "--override"),
    max_entries_per_config: int | None = typer.Option(
        None, "-m", "--max-entries-per-config"
    ),
):
    generate_predictions(
        max_entries_per_config=max_entries_per_config, override=override
    )
    evaluate_all()


@app.command()
def redact(
    text: str,
    config_path: str = typer.Option(CONFIG_FILE_NAME, "-c", "--config-path"),
):
    model = _get_model(config_path=config_path)
    print(redact_(text=text, model=model))


@app.command()
def identify(
    text: str,
    config_path: str = typer.Option(CONFIG_FILE_NAME, "-c", "--config-path"),
):
    model = _get_model(config_path=config_path)
    spans = model.predict_spans(text=text)
    print(json.dumps([s.model_dump() for s in spans], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
    port: int = typer.Option(8000, "--port", help="Port to bind to."),
    reload: bool = typer.Option(True, "--no-reload", help="Disable auto-reload."),
):
    """
    Start an API server exposing /redact and /identify endpoints.

    Both endpoints accept an optional "config" field (with the same shape as
    pii_eval.yaml) to override the generative-model settings on a per-request
    basis.
    """
    import uvicorn

    uvicorn.run("pii_eval.api:app", host=host, port=port, reload=reload)
