from ast import List
from copy import deepcopy
import json
from pathlib import Path
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
import spacy
from tqdm import tqdm

from dfm_anonymization.util.misc import DATASET_LABEL_MAP
from dfm_anonymization.util.util import presidio_result_2_span

OVERRIDE = False


class CustomPresidioModel():
    def __init__(self):
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "da", "model_name": "da_core_news_trf"}],
        }
        spacy.require_gpu(0)
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine_da = provider.create_engine()

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine_da,
            supported_languages=["da"]
        )

    def __call__(self, text: str, language="da") -> dict:
        presidio_result = self.analyzer.analyze(text, language=language)
        return [presidio_result_2_span(r, text) for r in presidio_result]


def main():
    model = CustomPresidioModel()

    for dataset_name in DATASET_LABEL_MAP.keys():

        print(dataset_name)
        input_fp = Path(f"data/span/{dataset_name}.json").resolve()
        output_path = Path(
            f"data/infer/{dataset_name}/custom_presidio.json").resolve()
        if output_path.exists() and not OVERRIDE:
            print("output exists, skipping")
            continue

        dataset: List[dict] = json.loads(input_fp.read_text(encoding="utf8"))

        results = []
        for entry in tqdm(dataset, dynamic_ncols=True):
            predicted_spans = model(entry["text"])
            predicted_entry = deepcopy(entry)
            predicted_entry["spans"] = predicted_spans
            results.append(predicted_entry)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=4), encoding="utf8")


if __name__ == "__main__":
    main()

pass
