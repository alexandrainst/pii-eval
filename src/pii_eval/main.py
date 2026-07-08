import typer

from pii_eval.evaluate import evaluate_all
from pii_eval.inference import generate_predictions

app = typer.Typer()


@app.command()
def run_evaluation():
    generate_predictions()
    evaluate_all()


@app.command()
def anonymize():
    raise NotImplementedError


if __name__ == "__main__":
    app()
