import json
from pathlib import Path
from typing import Dict, Optional
from gliner import GLiNER

from dfm_anonymization.evaluation.util import LABEL_TRANSLATIONS

dataset_name = "scrambletron_gen/da_cpr"
model_name = "nvidia/gliner-pii" #"urchade/gliner_multi-v2.1" #

dataset = json.loads(Path(f"data/span/{dataset_name}.json").read_text())
orig_labels = list(set([x["label"] for entry in dataset for x in entry["spans"]]))

labels = [k for k in LABEL_TRANSLATIONS[dataset_name].values() if k]

model = GLiNER.from_pretrained(model_name, map_location="cuda")

for entry in dataset:
    text = entry["full_text"]

    entities = model.predict_entities(text, labels)

    print(f"-- \"{text}\" --")
    if not entities:
        print("none found")
    else:
        for entity in entities:
            print(entity["text"], "=>", entity["label"])

        pass

