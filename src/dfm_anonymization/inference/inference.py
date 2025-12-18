from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, List

from tqdm import tqdm

from dfm_anonymization.inference.dataset import PiiDataset
from dfm_anonymization.inference.gliner_model import GLINER_MODEL_EXAMPLES, GlinerModel
from dfm_anonymization.inference.interfaces import PiiModel
from dfm_anonymization.inference.ner_model import NerModel
from dfm_anonymization.inference.presidio_model import PresidioModel
from dfm_anonymization.util.misc import LABEL_DESCRIPTIONS_DA


def inference(
        model: PiiModel, dataset: PiiDataset, show_bar: bool = True
) -> PiiDataset:
    dataset = deepcopy(dataset)
    it = tqdm(dataset) if show_bar else dataset
    for entry in it:
        timestamp = datetime.now().isoformat()
        entry.bio_tags = model(entry)
        entry.metadata = {
            "dataset_name": dataset.name,
            "model_name": model.name,
            "timestamp": timestamp
        }
    return dataset


def convert_datasets_from_words():
    for in_path in Path("data/words").glob("*.json"):
        dataset = PiiDataset.from_list_format_file(in_path)
        out_path = Path(f"data/ground_truth/{in_path.name}")
        dataset.to_file(out_path)


def generate_inferences():
    models: List[Callable[[], PiiModel]] = [
        # lambda: NerModel.from_hf("saattrupdan/nbailab-base-ner-scandi")
        # lambda: GlinerModel(GLINER_MODEL_EXAMPLES[0], LABEL_DESCRIPTIONS_DA),
        # lambda: GlinerModel(GLINER_MODEL_EXAMPLES[1], LABEL_DESCRIPTIONS_DA)
        lambda: PresidioModel()
    ]
    for model_fct in models:
        model = model_fct()
        print(f"Model: {model.name}")
        for in_path in Path("data/ground_truth").glob("*.json"):
            dataset = PiiDataset.from_file(in_path)
            print(f"Dataset: {dataset.name}")
            predictions = inference(model=model, dataset=dataset)
            predictions.to_file(
                f"data/infer_ds/{model.name}/{dataset.name}.json")


if __name__ == "__main__":
    # convert_datasets_from_words()
    generate_inferences()

    pass


# sample_ds = PiiDataset.sample()
# sample_json = sample_ds.to_json()
# sample_ds_ = PiiDataset.from_json(sample_json)
# assert sample_ds == sample_ds_
