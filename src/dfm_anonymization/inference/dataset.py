import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dfm_anonymization.util.util import to_dict_recursive


@dataclass
class PiiDataEntry(object):
    id: int | str
    text: str
    tokens: List[str]
    bio_tags: Optional[List[str]] = None
    metadata: Optional[dict] = None

    @staticmethod
    def sample() -> "PiiDataEntry":
        return PiiDataEntry(
            id=0,
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
        for k in meta.keys():
            if k.startswith('__'):
                _ = meta.pop(k)
        sd['metadata'] = meta
        return PiiDataEntry(**sd)


@dataclass
class PiiDataset(object):
    name: str
    entries: List[PiiDataEntry]
    metadata: Optional[dict] = None

    def __iter__(self):
        return self.entries.__iter__()

    def __len__(self):
        return len(self.entries)

    def labels(self):
        return {t[2:] for e in self.entries for t in e.bio_tags if t != "O"}

    @staticmethod
    def sample() -> "PiiDataset":
        return PiiDataset("sample_dataset", entries=[PiiDataEntry.sample()])

    @staticmethod
    def from_file(file_path: Path | str) -> "PiiDataset":
        if type(file_path) == str:
            file_path = Path(file_path)
        return PiiDataset.from_json_str(file_path.read_text(encoding="utf8"))

    @staticmethod
    def from_json_str(json_str) -> "PiiDataset":
        return PiiDataset.from_dict(json.loads(json_str))

    @staticmethod
    def from_dict(d: dict) -> "PiiDataset":
        entries = []
        for i, e in enumerate(d['entries']):
            if 'id' not in e:
                e['id'] = i
            entries.append(PiiDataEntry.from_dict(e))
        d['entries'] = entries
        original_metadata = d['metadata'] if 'metadata' in d else dict()
        metadata = original_metadata | {
            k: v for k, v in d.items()
            if k not in dir(PiiDataset)
        }
        for k in metadata.keys():
            if k.startswith('__'):
                _ = metadata.pop(k)
        d['metadata'] = metadata
        return PiiDataset(**d)

    def to_json_str(self) -> str:
        return json.dumps(to_dict_recursive(self), ensure_ascii=False, indent=2)

    def to_file(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.write_text(self.to_json_str(), encoding="utf8")

    @staticmethod
    def from_list_format_file(
            file: Path | str, dataset_name: str | None = None
    ) -> "PiiDataset":
        if isinstance(file, str):
            file = Path(file)
        assert file.exists()
        cnt: List[dict] = json.loads(file.read_text(encoding="utf8"))
        entries = []
        for entry in cnt:
            entry['tokens'] = entry.pop('words')
            entry['bio_tags'] = entry.pop('true_bio_tags')
            entry['id'] = entry.pop('document')
            entries.append(PiiDataEntry.from_dict(entry))
        return PiiDataset(
            name=dataset_name if dataset_name else file.stem,
            entries=entries,
            metadata={
                "file_path": str(file)
            }
        )
