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
    text_or_path: str,
    whitelist: Optional[List[str]] = typer.Option(
        None, "--whitelist", help="Whitelist - strings to avoid anonymizing"
    ),
):
    maybe_path = Path(text_or_path)
    if maybe_path.exists():
        out_path = maybe_path.parent / \
            f"{maybe_path.stem}-anonymized-{maybe_path.suffix}"
        text = maybe_path.read_text(encoding="utf8")
    else:
        text = text_or_path

    anonymizer = Anonymizer()
    result = anonymizer(text, whitelist)
    if out_path:
        out_path.write_text(result.text, encoding="utf8")
    else:
        print(result.text)



def main() -> int:
    return app()
