from datetime import datetime
from pathlib import Path

from codetiming import Timer
from dotenv import load_dotenv
from rich.progress import MofNCompleteColumn, Progress

from dfm_anonymization.constants import LABEL_DESCRIPTIONS_DA
from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.models.all_models import *


def inference(
    model: PiiModel,
    dataset: PiiDataset,
    samples: int | None = None,
    progress: Progress | None = None,
) -> PiiDataset:
    it = dataset.entries if samples is None else dataset.get_samples(samples)
    new_entries = []
    entries_task = progress.add_task("", total=len(it))
    for entry in it:
        progress.update(
            entries_task,
            advance=1,
            description=f"Entry: {str(entry.id)[:10]:<10}",
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
        OpenAIClientModel(
            model_tag="qwen3.5-397b",
            base_url="https://inference.alexandra.dk/v1",
            api_key=os.getenv("ALEX_API_KEY"),
            name="qwen3.5-397b",
        ),
        NerModel.from_hf(tag="tabularisai/eu-pii-safeguard", out_bias=x)
        for x in [-1, -2, -3, -4, -6, -7, -8, -9]
    ] + [
        GlinerModel("nvidia/gliner-pii", LABEL_DESCRIPTIONS_DA, threshold=x)
        for x in [0.5, 0.2, 0.05, 0.02, 0.005, 0.002]
    ]

    datasets = list(Path("data/ground_truth").glob("*.json"))

    print(
        f"generating predictions\nmodels: {', '.join([m.name for m in models])}\n"
        f"datasets: {', '.join([d.stem for d in datasets])}"
    )

    progress = Progress(*Progress.get_default_columns(), MofNCompleteColumn())
    progress.start()
    model_task = progress.add_task("", total=len(models), transient=True)
    while models:
        model = models.pop(0)
        progress.update(
            model_task,
            advance=1,
            description=f"Model: {model.name[:10]:<10}",
        )
        dataset_task = progress.add_task("", total=len(datasets), transient=True)
        for in_path in datasets:
            dataset = PiiDataset.from_file(in_path)
            dataset_name = dataset.metadata.get("dataset_name", in_path.stem)
            progress.update(
                dataset_task,
                advance=1,
                description=f"Dataset: {dataset_name[:10]:<10}",
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
    progress.stop()


if __name__ == "__main__":
    generate_predictions(override=False)
