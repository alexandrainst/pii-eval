from copy import deepcopy
import itertools
import json
from pathlib import Path
from gliner import GLiNER
from tqdm import tqdm

from dfm_anonymization.util.misc import DATASET_LABEL_MAP

OVERRIDE = False

datasets = list(DATASET_LABEL_MAP.keys())
models = ["nvidia/gliner-pii", "urchade/gliner_multi-v2.1"]

it = list(itertools.product(datasets, models))

i = 0
for dataset_name, model_name in it:
    i += 1
    print(f"dataset: {dataset_name}, model: {model_name} ({i} / {len(it)})")
    in_path = Path(f"data/span/{dataset_name}.json")
    out_path = Path(f"data/infer/{dataset_name}/{model_name.replace('/', '-')}.json")
    if not OVERRIDE and out_path.exists():
        print("skipping, output exists")
        continue
    dataset = json.loads(in_path.read_text())

    label_translations = DATASET_LABEL_MAP[dataset_name]
    label2orig = {v: k for k, v in label_translations.items() if v}
    labels = list(label2orig.keys())

    model = GLiNER.from_pretrained(model_name, map_location="cuda")

    new_entries = []
    for entry in tqdm(dataset, dynamic_ncols=True):
        text: str = entry["text"]
        spans = model.predict_entities(text, labels)
        for span in spans:
            span["label"] = label2orig[span["label"]]

        new_entry = deepcopy(entry)
        new_entry["spans"] = spans
        new_entries.append(new_entry)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(new_entries, ensure_ascii=False, indent=4))

