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

model: PiiModel = Regex()
config = load_config()
gen_config = config.generative
if gen_config.active:
    if not gen_config.is_ready():
        print(
            f"generative.active=true in {CONFIG_FILE_NAME} but generative.model "
            "or generative.base_url is missing. "
            "Falling back to regex based recognition."
        )
    else:
        model = Ensemble.default()


@app.command()
def run_evaluation():
    generate_predictions()
    evaluate_all()


@app.command()
def redact(text: str):
    print(redact_(text=text, model=model))


@app.command()
def identify(text: str):
    spans = model.predict_spans(text=text)
    print(json.dumps([asdict(s) for s in spans], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
