from dataclasses import asdict
import json

import typer

from pii_eval.config import CONFIG_FILE_NAME, load_config
from pii_eval.data import PiiModel
from pii_eval.evaluate import evaluate_all
from pii_eval.inference import generate_predictions
from pii_eval.models.ensemble import Ensemble
from pii_eval.models.regex_model import Regex
from pii_eval.redact import redact as redact_

app = typer.Typer()


def _get_model(config_path: str = "./pii_eval_config.yaml"):
    model: PiiModel = Regex()
    config = load_config(config_path)
    gen_config = config.generative
    if gen_config.active:
        if not gen_config.is_ready():
            print(
                f"generative.active=true in {CONFIG_FILE_NAME} but "
                "failed to contact api. "
                "Falling back to regex based recognition."
            )
        else:
            model = Ensemble.default()
    return model


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
    config_path: str = typer.Option("./pii_eval_config.yaml", "-c", "--config-path"),
):
    model = _get_model(config_path=config_path)
    print(redact_(text=text, model=model))


@app.command()
def identify(
    text: str,
    config_path: str = typer.Option("./pii_eval_config.yaml", "-c", "--config-path"),
):
    model = _get_model(config_path=config_path)
    spans = model.predict_spans(text=text)
    print(json.dumps([asdict(s) for s in spans], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    model = _get_model(config_path="./pii_eval_config.yaml")
    app()
