from datetime import datetime
import itertools
from pathlib import Path
from typing import Callable, List, Tuple

from codetiming import Timer
from tqdm import tqdm

from dfm_anonymization.constants import LABEL_DESCRIPTIONS_DA
from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.models.ensemble_model import EnsembleModel
from dfm_anonymization.models.gliner_model import GlinerModel
from dfm_anonymization.models.ner_model import NerModel
from dfm_anonymization.models.presidio_model import PresidioModel
from dfm_anonymization.models.scrambletron import ScrambletronModel


def inference(
    model: PiiModel,
    dataset: PiiDataset,
    show_progress_bar: bool = True,
    max_entries: int = None,
) -> PiiDataset:
    it = tqdm(dataset, dynamic_ncols=True) if show_progress_bar else dataset
    if max_entries:
        it = itertools.islice(it, max_entries)
    new_entries = []
    for entry in it:
        new_entries.append(
            PiiDataEntry(
                id=entry.id,
                text=entry.text,
                tokens=entry.tokens,
                bio_tags=model(entry),
                metadata=entry.metadata,
            )
        )
    timestamp = datetime.now().isoformat()
    return PiiDataset(
        entries=new_entries,
        metadata={
            "model_name": model.name,
            "is_ground_truth": False,
            "timestamp": timestamp,
            "dataset": dataset.metadata,
        },
    )


def convert_datasets_from_words():
    for in_path in Path("data/words").glob("*.json"):
        dataset = PiiDataset.from_list_format_file(in_path)
        out_path = Path(f"data/ground_truth/{in_path.name}")
        dataset.to_file(out_path)


def generate_predictions(override: bool = False):
    models: List[Tuple[str, Callable[[], PiiModel]]] = [
        (
            "scandiner",
            lambda: NerModel.from_hf("saattrupdan/nbailab-base-ner-scandi"),
        ),
        (
            "gliner-pii",
            lambda: GlinerModel("nvidia/gliner-pii", LABEL_DESCRIPTIONS_DA),
        ),
        (
            "gliner_multi-v2.1",
            lambda: GlinerModel("urchade/gliner_multi-v2.1", LABEL_DESCRIPTIONS_DA),
        ),
        ("presidio", lambda: PresidioModel()),
        ("ensemble_gliner_pii_presidio", EnsembleModel.get_gliner_pii_presidio),
        ("scramletron", ScrambletronModel),
    ]
    datasets = list(Path("data/ground_truth").glob("*.json"))
    print(
        f"generating predictions\nmodels: {', '.join([m[0] for m in models])}\n"
        f"datasets: {', '.join([d.stem for d in datasets])}"
    )

    for model_name, model_fct in models:
        model = None
        print(f"Model: {model_name}")
        for in_path in datasets:
            dataset = PiiDataset.from_file(in_path)
            dataset_name = dataset.metadata.get("dataset_name", in_path.stem)
            print(f"Dataset: {dataset_name}")
            output_file_path = Path(
                f"data/predictions/{model_name}/{dataset_name}.json"
            )
            if not override and output_file_path.exists():
                print("skipping: file already exists")
                continue
            if not model:
                model = model_fct()
            timer = Timer()
            timer.start()
            predictions = inference(model=model, dataset=dataset, max_entries=200)
            timer.stop()
            elapsed_time = timer.last
            predictions.metadata.update(
                {
                    "computation_time_total": elapsed_time,
                    "computation_time_per_entry": elapsed_time / len(dataset),
                }
            )
            predictions.to_file(output_file_path)
    print()


if __name__ == "__main__":
    generate_predictions(override=False)
