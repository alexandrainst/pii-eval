import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification

from dfm_anonymization.util.util import to_dict_recursive


@dataclass
class PiiDataEntry(object):
    text: str
    tokens: List[str]
    bio_tags: Optional[List[str]] = None
    metadata: Optional[dict] = None

    @staticmethod
    def sample() -> "PiiDataEntry":
        return PiiDataEntry(
            text="\"Hør du nu, jomfru Hermelin,",
            tokens=["\"", "Hør", "du", "nu", ",", "jomfru", "Hermelin", ","],
            bio_tags=["O", "O", "O", "O", "O", "O", "B-PERSON", "O"]
        )

    @staticmethod
    def from_dict(d: dict) -> "PiiDataEntry":
        attribute_keys = PiiDataEntry.__annotations__.keys()
        sd = {k: v for k, v in d.items() if k in attribute_keys}
        meta = {k: v for k, v in d.items() if k not in attribute_keys}
        if 'metadata' in d:
            meta = meta | d['metadata']
        sd['metadata'] = meta
        return PiiDataEntry(**sd)


@dataclass
class PiiDataset(object):
    name: str
    entries: List[PiiDataEntry]
    metadata: Optional[dict] = None

    def __iter__(self):
        return self.entries.__iter__()

    @staticmethod
    def sample() -> "PiiDataset":
        return PiiDataset("sample_dataset", entries=[PiiDataEntry.sample()])

    @staticmethod
    def from_json_str(json_str) -> "PiiDataset":
        return PiiDataset.from_dict(json.loads(json_str))

    @staticmethod
    def from_dict(d: dict) -> "PiiDataset":
        d['entries'] = [PiiDataEntry.from_dict(e) for e in d['entries']]
        original_metadata = d['metadata'] if 'metadata' in d else dict()
        d['metadata'] = original_metadata | {
            k: v for k, v in d.items()
            if k not in dir(PiiDataset)  # ["name", "entries", "metadata"]
        }
        return PiiDataset(**d)

    def to_json_str(self) -> str:
        return json.dumps(to_dict_recursive(self), ensure_ascii=False)


def from_list_format_file(
        file: Path | str, dataset_name: str | None = None
) -> PiiDataset:
    if isinstance(file, str):
        file = Path(file)
    assert file.exists()
    cnt = json.loads(file.read_text(encoding="utf8"))
    entries = []
    for entry in cnt:
        entry['tokens'] = entry.pop('words')
        entries.append(PiiDataEntry.from_dict(entry))
    return PiiDataset(
        name=dataset_name if dataset_name else file.stem,
        entries=entries,
        metadata={
            "file_path": str(file)
        }
    )


class Tokenizer(ABC, Callable):
    @abstractmethod
    def __call__(self, words: List[str]) -> Tuple[Any, List[int]]:
        pass


class TokenClassifier(ABC, Callable):
    @abstractmethod
    def __call__(self, tokens: Any) -> List[str]:
        pass


class PiiModel(ABC, Callable):
    name: str

    @abstractmethod
    def __call__(self, tokens: List[str]) -> List[str]:
        pass


class NerModel(ABC, PiiModel):
    def __init__(self, name: str, tokenizer: Tokenizer, model: TokenClassifier):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, words: List[str]) -> List[str]:
        subtokens, word_ids = self.tokenizer(words)
        sublabels = self.model(subtokens)
        labels = []
        current_id = 0
        for sublabel, word_id in zip(sublabels, word_ids):
            if current_id > word_id:
                continue
            if sublabel.startswith("B-"):
                labels.append(sublabel)
                current_id += 1
                continue
            elif sublabel.startswith("I-"):
                if not labels or labels[-1][2:] != sublabel[2:]:
                    labels.append("B-" + sublabel[2:])
                else:
                    labels.append(sublabel)
                current_id += 1
                continue
            else:
                continue
        return labels


def inference(model: PiiModel, dataset: PiiDataset) -> PiiDataset:
    dataset = deepcopy(dataset)
    for entry in dataset:
        timestamp = datetime.now().isoformat()
        entry.bio_tags = model(entry.tokens)
        entry.metadata = {
            "dataset_name": dataset.name,
            "model_name": model.name,
            "timestamp": timestamp
        }
    return dataset


model_name = "saattrupdan/nbailab-base-ner-scandi"
dataset_name = "gen_address"
dataset = from_list_format_file(f"data/words/{dataset_name}.json")
model = NerModel(
    name=model_name,
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    mode=AutoModelForTokenClassification.from_pretrained(model_name)
)

predictions = inference(model=model, dataset=dataset)

pass


# sample_ds = PiiDataset.sample()
# sample_json = sample_ds.to_json()
# sample_ds_ = PiiDataset.from_json(sample_json)
# assert sample_ds == sample_ds_
