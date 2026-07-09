import os

import typer
from dotenv import load_dotenv

from pii_eval.evaluate import evaluate_all
from pii_eval.inference import generate_predictions
from pii_eval.models.ensemble import Ensemble
from pii_eval.models.regex_model import Regex
from pii_eval.redact import redact as redact_

app = typer.Typer()

dot_env_path = ".env.pii_eval"
load_dotenv(dot_env_path)
gen_active: str = os.environ.get("GEN_ACTIVE", None) == "1"
model_tag: str = os.environ.get("GEN_MODEL", None)
base_url: str = os.environ.get("GEN_BASE_URL", None)
api_key: str = os.environ.get("GEN_API_KEY", None)

if gen_active:
    if not base_url or not model_tag:
        print(
            f"GEN_ACTIVE=1 in {dot_env_path} but GEN_MODEL or GEN_BASE_URL is missing. "
            "Falling back to regex based recognition."
        )
        model = Regex()
    else:
        model = Ensemble.default()


@app.command()
def run_evaluation():
    generate_predictions()
    evaluate_all()


@app.command()
def redact(text: str):
    print(redact_(text=text, model=model))


if __name__ == "__main__":
    app()
