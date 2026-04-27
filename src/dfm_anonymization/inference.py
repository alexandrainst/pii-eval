import os
from datetime import datetime
from pathlib import Path

from codetiming import Timer
from dotenv import load_dotenv
from tqdm import tqdm
from rich.progress import track

from dfm_anonymization.constants import (
    ALL_ISO_LABELS,
    LABEL_DESCRIPTIONS_DA,
    MODEL_LABEL_MAP,
)
from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.models.gliner_model import GlinerModel
from dfm_anonymization.models.ner_model import NerModel
from dfm_anonymization.models.openai_client_model import OpenAIClientModel
from dfm_anonymization.models.presidio_model import PresidioModel
from dfm_anonymization.models.regex_model import RegexModel
from dfm_anonymization.models.ensemble_model import EnsembleModel
from dfm_anonymization.util.util import IdenMap
# from dfm_anonymization.models.scrambletron import ScrambletronModel


def inference(
    model: PiiModel,
    dataset: PiiDataset,
    samples: int | None = None,
    # progress: Progress | None = None,  # pbar_pos: int = 0
    # task_id: TaskID | None = None,
) -> PiiDataset:
    it = track(
        dataset if samples is None else dataset.get_samples(samples),
        description="Entries...",
        transient=True,
    )
    new_entries = []
    for entry in it:
        try:
            prediction = model(entry)
        except Exception as exc:
            entry_id = entry.id if entry.id is not None else "<unknown>"
            raise RuntimeError(
                f"Inference failed for model '{model.name}' on entry '{entry_id}'."
            ) from exc
        new_entries.append(
            PiiDataEntry(
                id=entry.id,
                text=entry.text,
                tokens=entry.tokens,
                bio_tags=prediction,
                metadata=entry.metadata,
            )
        )
    # progress.remove_task(entries_task)
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


def generate_predictions(
    override: bool = False, max_entries_per_config: int | None = None, pbar_pos: int = 0
):
    load_dotenv()

    models: list[PiiModel] = [
        NerModel.from_hf(tag="tabularisai/eu-pii-safeguard", out_bias=-10.0),
        # NerModel.from_hf(tag="saattrupdan/nbailab-base-ner-scandi", out_bias=20.0 - 5.0 * x)
        # NerModel.from_hf(tag="openai/privacy-filter") # , out_bias=20.0 - 5.0 * x)
        # PresidioModel(),
        EnsembleModel(
            name="ensemble_ner",
            models=[
                (PresidioModel(), MODEL_LABEL_MAP["presidio"]),
                (
                    GlinerModel(
                        "nvidia/gliner-pii", LABEL_DESCRIPTIONS_DA, threshold=0.01
                    ),
                    {k: k for k in ALL_ISO_LABELS},
                ),
                (
                    NerModel.from_hf("tabularisai/eu-pii-safeguard", out_bias=-10.0),
                    MODEL_LABEL_MAP["eu-pii-safeguard*"],
                ),
                (
                    NerModel.from_hf(
                        "saattrupdan/nbailab-base-ner-scandi", out_bias=-5.0
                    ),
                    MODEL_LABEL_MAP["nbailab-base-ner-scandi"],
                ),
            ],
        )
        # RegexModel(),
        # NerModel.from_hf("saattrupdan/nbailab-base-ner-scandi"),
        # GlinerModel("nvidia/gliner-pii", LABEL_DESCRIPTIONS_DA, threshold=0.01)
        # GlinerModel("urchade/gliner_multi-v2.1", LABEL_DESCRIPTIONS_DA),
        # OpenAIClientModel(
        #     model_tag="Qwen/Qwen3.5-4B",
        #     base_url="http://localhost:8000/v1",
        #     api_key="not needed",
        # ),
        # OpenAIClientModel(
        #     model_tag="qwen-235b",
        #     base_url="https://inference.projects.alexandrainst.dk/v1",
        #     api_key=os.getenv("ALEXANDRA_INFERENCE_API_KEY"),
        # ),
        # OpenAIClientModel(
        #     model_tag="Qwen/Qwen3.5-27B",
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=os.getenv("OPENROUTER_API_KEY"),
        # ),
        # ScrambletronModel(),
        # EnsembleModel.get_gliner_pii_regex()
    ]
    datasets = list(Path("data/ground_truth").glob("*.json"))
    # datasets = [Path("data/ground_truth/openpii-1m.json")]

    print(
        f"generating predictions\nmodels: {', '.join([m.name for m in models])}\n"
        f"datasets: {', '.join([d.stem for d in datasets])}"
    )

    # progress = Progress()
    # progress.start()
    # model_task = progress.add_task("Models...", total=len(models))
    # dataset_task = progress.add_task("Datasets...", total=len(datasets))
    for model in track(models):
        track.columns[0].text = f"Model: {model.name}"
        # progress.advance(model_task)
        # progress.reset(dataset_task)
        for in_path in track(datasets):
            # progress.advance(dataset_task)
            dataset = PiiDataset.from_file(in_path)
            dataset_name = dataset.metadata.get("dataset_name", in_path.stem)
            track.columns[1].text = f"Dataset: {dataset.name}"
            output_file_path = Path(
                f"data/predictions/{model.name}/{dataset_name}.json"
            )
            if not override and output_file_path.exists():
                print(f"skipped {dataset_name} (was cached)")
            else:
                timer = Timer(logger=None)
                timer.start()
                predictions = inference(
                    model=model,
                    dataset=dataset,
                    samples=max_entries_per_config,
                    # progress=progress,
                    # task_id=None,
                )
                timer.stop()
                elapsed_time = timer.last
                predictions.metadata.update(
                    {
                        "computation_time_total": elapsed_time,
                        "computation_time_per_entry": elapsed_time / len(predictions),
                    }
                )
                predictions.to_file(output_file_path)
    # progress.stop()


if __name__ == "__main__":
    generate_predictions(
        override=False,
        # max_entries_per_config=500,
    )
