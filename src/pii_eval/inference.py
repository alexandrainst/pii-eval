from datetime import datetime
from pathlib import Path

from codetiming import Timer
from dotenv import load_dotenv
from rich.progress import MofNCompleteColumn, Progress

from pii_eval.data import PiiDataset, PiiModel
from pii_eval.models.regex_model import RegexModel


def inference(
    model: PiiModel,
    dataset: PiiDataset,
    samples: int | None = None,
    progress: Progress | None = None,
) -> PiiDataset:
    it = dataset.entries if samples is None else dataset.get_samples(samples)
    predictions = []
    entries_task = progress.add_task("", total=len(it))
    for entry in it:
        progress.update(
            entries_task,
            advance=1,
            description=f"Entry: {str(entry.uid)[:10]:<10}",
            transient=True,
        )
        try:
            prediction = model(entry)
        except Exception as exc:
            entry_id = entry.uid if entry.uid is not None else "<unknown>"
            raise RuntimeError(
                f"Inference failed for model '{model.name}' on entry '{entry_id}'."
            ) from exc
        predictions.append(prediction)
    progress.remove_task(entries_task)
    timestamp = datetime.now().isoformat()
    return PiiDataset(
        name=dataset.name,
        entries=predictions,
        meta={
            "model_name": model.name,
            "timestamp": timestamp,
            "dataset": dataset.meta,
        },
    )


def generate_predictions(
    override: bool = False, max_entries_per_config: int | None = None, pbar_pos: int = 0
):
    load_dotenv()

    models: list[PiiModel] = [RegexModel()]

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
            dataset = PiiDataset.load_json(in_path)
            dataset_name = dataset.meta.get("dataset_name", in_path.stem)
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
                )
                timer.stop()
                elapsed_time = timer.last
                predictions.meta.update(
                    {
                        "computation_time_total": elapsed_time,
                        "computation_time_per_entry": elapsed_time / len(predictions),
                    }
                )
                predictions.save_json(output_file_path)
        progress.remove_task(dataset_task)
    progress.remove_task(model_task)


if __name__ == "__main__":
    generate_predictions(override=False)
