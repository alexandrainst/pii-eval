import os
from datetime import datetime
from pathlib import Path

from codetiming import Timer
from dotenv import load_dotenv
from rich.progress import Progress, track

from dfm_anonymization.constants import (
    ALL_ISO_LABELS,
    LABEL_DESCRIPTIONS_DA,
    MODEL_LABEL_MAP,
)
from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.models.gliner_model import GlinerModel
from dfm_anonymization.models.ner_model import NerModel
from dfm_anonymization.models.opf_model import OpfNerModel
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
    progress: Progress | None = None,  # pbar_pos: int = 0
    # task_id: TaskID | None = None,
) -> PiiDataset:
    it = dataset.entries if samples is None else dataset.get_samples(samples)
    new_entries = []
    entries_task = progress.add_task("", total=len(it))
    for entry in it:
        progress.update(
            entries_task,
            advance=1,
            description=f"Entry: {entry.id[:10]:<10}",
            transient=True,
        )
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
    progress.remove_task(entries_task)
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
        NerModel.from_hf(
            tag="tabularisai/eu-pii-safeguard", out_bias=-60.0
        )  # , out_bias=5.0 - 5.0 * x)
        # for x in range(10)
    ]
    # + [
    #     GlinerModel(
    #         "nvidia/gliner-pii", LABEL_DESCRIPTIONS_DA, threshold=0.01 - 0.001 * x
    #     )
    #     for x in range(10)
    # ]
    datasets = list(Path("data/ground_truth").glob("*.json"))
    # datasets = [Path("data/ground_truth/openpii-1m.json")]
    # datasets = [Path("data/ground_truth/dimmi.json")]

    print(
        f"generating predictions\nmodels: {', '.join([m.name for m in models])}\n"
        f"datasets: {', '.join([d.stem for d in datasets])}"
    )

    progress = Progress()
    progress.start()
    model_task = progress.add_task("", total=len(models))
    while models:
        model = models.pop(0)
        progress.update(
            model_task,
            advance=1,
            description=f"Model: {model.name[:10]:<10}",
            transient=True,
        )
        dataset_task = progress.add_task("", total=len(datasets))
        for in_path in datasets:
            dataset = PiiDataset.from_file(in_path)
            dataset_name = dataset.metadata.get("dataset_name", in_path.stem)
            progress.update(
                dataset_task,
                advance=1,
                description=f"Dataset: {dataset_name[:10]:<10}",
                transient=True,
            )
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
                    progress=progress,
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
        progress.remove_task(dataset_task)
    progress.remove_task(model_task)
    # print()
    # progress.stop()


if __name__ == "__main__":
    generate_predictions(
        override=False,
        # max_entries_per_config=500,
    )
