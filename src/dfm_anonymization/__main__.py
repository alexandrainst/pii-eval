import sys
from pathlib import Path
from typing import List, Optional

import typer

from dfm_anonymization.anonymize import Anonymizer
from dfm_anonymization.evaluate import evaluate_all
from dfm_anonymization.inference import generate_predictions
from dfm_anonymization.load import load_datasets

app = typer.Typer()


@app.command(help="Make and show evaluations.")
def make_evaluations():
    load_datasets()
    generate_predictions()
    evaluate_all()


@app.command(
    help="Anonymize text.\n\n Example:\ndfm-anonymization anonymize "
    '"Navnet er Anders Sand. Og 21324354 er telefonnummeret." '
    '--whitelist "Anders Sand"'
)
def anonymize(
    text_or_path: str | None = typer.Argument(
        None, help="A string or text file to anonymize."
    ),
    whitelist: Optional[List[str]] = typer.Option(
        None, "--whitelist", help="Whitelist - strings to avoid anonymizing"
    ),
):
    if text_or_path is None or text_or_path == '-':
        text_or_path = sys.stdin.read().strip()
    maybe_path = Path(text_or_path)
    if maybe_path.exists():
        text = maybe_path.read_text(encoding="utf8")
    else:
        text = text_or_path
    anonymizer = Anonymizer()
    result = anonymizer(text, whitelist)
    print(result.text)


def main() -> int:
    return app()
