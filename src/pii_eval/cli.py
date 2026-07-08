import typer

from pii_eval.models.regex_model import Regex
from pii_eval.redact import redact as redact_
from pii_eval.evaluate import evaluate_all
from pii_eval.inference import generate_predictions

app = typer.Typer()


@app.command()
def run_evaluation():
    generate_predictions()
    evaluate_all()


@app.command()
def redact(text: str):
    print(redact_(text=text, model=Regex()))


if __name__ == "__main__":
    app()
