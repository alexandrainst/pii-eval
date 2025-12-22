from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, List

from codetiming import Timer
from tqdm import tqdm

from dfm_anonymization.inference.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.inference.ensemble_model import EnsembleModel
from dfm_anonymization.inference.gliner_model import (
    GLINER_MODEL_EXAMPLES, GlinerModel)
from dfm_anonymization.inference.interfaces import PiiModel
from dfm_anonymization.inference.ner_model import NerModel
from dfm_anonymization.inference.presidio_model import PresidioModel
from dfm_anonymization.util.misc import LABEL_DESCRIPTIONS_DA


def inference(
        model: PiiModel, dataset: PiiDataset, show_progress_bar: bool = True
) -> PiiDataset:
    it = tqdm(dataset, dynamic_ncols=True) if show_progress_bar else dataset
    new_entries = []
    for entry in it:
        new_entries.append(PiiDataEntry(
            id=entry.id,
            text=entry.text,
            tokens=entry.tokens,
            bio_tags=model(entry),
            metadata=entry.metadata
        ))
    timestamp = datetime.now().isoformat()
    return PiiDataset(entries=new_entries, metadata={
        "model_name": model.name,
        "is_ground_truth": False,
        "timestamp": timestamp,
        "dataset": dataset.metadata
    })


def convert_datasets_from_words():
    for in_path in Path("data/words").glob("*.json"):
        dataset = PiiDataset.from_list_format_file(in_path)
        out_path = Path(f"data/ground_truth/{in_path.name}")
        dataset.to_file(out_path)


def generate_inferences(override: bool = False):
    models: List[Callable[[], PiiModel]] = [
        lambda: NerModel.from_hf("saattrupdan/nbailab-base-ner-scandi"),
        lambda: GlinerModel(GLINER_MODEL_EXAMPLES[0], LABEL_DESCRIPTIONS_DA),
        lambda: GlinerModel(GLINER_MODEL_EXAMPLES[1], LABEL_DESCRIPTIONS_DA),
        lambda: PresidioModel(),
        EnsembleModel.get_gliner_pii_presidio
    ]
    for model_fct in models:
        model = model_fct()
        print(f"Model: {model.name}")
        for in_path in Path("data/ground_truth").glob("*.json"):
            dataset = PiiDataset.from_file(in_path)
            dataset_name = dataset.metadata['dataset_name']
            print(f"Dataset: {dataset_name}")
            output_file_path = Path(
                f"data/infer_ds/{model.name}/{dataset_name}.json")
            if not override and output_file_path.exists():
                print("skipping: file already exists")
                continue
            timer = Timer()
            timer.start()
            predictions = inference(model=model, dataset=dataset)
            timer.stop()
            time = timer.last
            predictions.metadata.update({
                "computation_time_total": time,
                "computation_time_per_entry": time / len(dataset)
            })
            predictions.to_file(output_file_path)


if __name__ == "__main__":
    # convert_datasets_from_words()
    generate_inferences(override=True)

    pass
