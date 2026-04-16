from datetime import datetime
from pathlib import Path

from codetiming import Timer
from dotenv import load_dotenv
from tqdm import tqdm

from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.interfaces import PiiModel

# from dfm_anonymization.models.ensemble_model import EnsembleModel
from dfm_anonymization.models.regex_model import RegexModel
# from dfm_anonymization.models.gliner_model import GlinerModel
# from dfm_anonymization.models.ner_model import NerModel
# from dfm_anonymization.models.openai_client_model import OpenAIClientModel
# from dfm_anonymization.models.presidio_model import PresidioModel
# from dfm_anonymization.models.scrambletron import ScrambletronModel


def inference(
    model: PiiModel, dataset: PiiDataset, samples: int | None = None, pbar_pos: int = 0
) -> PiiDataset:
    it = dataset if samples is None else dataset.get_samples(samples)
    new_entries = []
    for entry in tqdm(it, dynamic_ncols=True, leave=False, position=pbar_pos):
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
        RegexModel(),
        # NerModel.from_hf("saattrupdan/nbailab-base-ner-scandi"),
        # PresidioModel(),
        # GlinerModel("nvidia/gliner-pii", LABEL_DESCRIPTIONS_DA),
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
    # datasets = list(Path("data/ground_truth").glob("*.json"))
    datasets = [Path("data/ground_truth/openpii-1m-da-s.json")]

    print(
        f"generating predictions\nmodels: {', '.join([m.name for m in models])}\n"
        f"datasets: {', '.join([d.stem for d in datasets])}"
    )

    for model in tqdm(models, leave=False, position=pbar_pos, dynamic_ncols=True):
        print(f"Model: {model.name}")
        for in_path in (
            pbar_datasets := tqdm(
                datasets, leave=False, position=pbar_pos + 1, dynamic_ncols=True
            )
        ):
            dataset = PiiDataset.from_file(in_path)
            dataset_name = dataset.metadata.get("dataset_name", in_path.stem)
            print(f"Dataset: {dataset_name}")
            output_file_path = Path(
                f"data/predictions/{model.name}/{dataset_name}.json"
            )
            if not override and output_file_path.exists():
                pbar_datasets.desc = f"skipped {dataset_name} (was cached)"
                continue
            timer = Timer()
            timer.start()
            predictions = inference(
                model=model,
                dataset=dataset,
                samples=max_entries_per_config,
                pbar_pos=pbar_pos + 2,
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
    print()


if __name__ == "__main__":
    generate_predictions(override=True)  # , max_entries_per_config=500)
