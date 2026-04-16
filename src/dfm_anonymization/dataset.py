from datetime import datetime
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from dfm_anonymization.util.spans import LabelSpan, Span, spans_to_bio
from dfm_anonymization.util.util import WordTokenizer, to_dict_recursive


@dataclass
class PiiDataEntry(object):
    id: int | str
    text: str
    tokens: list[str]
    token_spans: list[tuple[int, int]] | None = None
    bio_tags: list[str] | None = None
    metadata: dict | None = None

    def get_token_spans(self) -> list[tuple[int, int]]:
        if self.token_spans is None:
            self.token_spans = []
            i = 0
            for token in self.tokens:
                i = self.text.find(token, i)
                assert i >= 0
                self.token_spans.append((i, i + len(token)))
                i += len(token)
            assert i <= len(self.text)
        return self.token_spans

    @staticmethod
    def example() -> "PiiDataEntry":
        return PiiDataEntry(
            id=0,
            text='"Hør du nu, jomfru Hermelin,',
            tokens=['"', "Hør", "du", "nu", ",", "jomfru", "Hermelin", ","],
            bio_tags=["O", "O", "O", "O", "O", "O", "B-PERSON", "O"],
        )

    @staticmethod
    def from_txt_file(path: Path | str) -> "PiiDataEntry":
        if type(path) == str:
            path = Path(path)
        assert path.exists()
        text = path.read_text(encoding="utf8")
        word_tokenizer = WordTokenizer()
        words, word_spans = word_tokenizer.word_tokenize(text=text)
        return PiiDataEntry(text=text, tokens=words, token_spans=word_spans)

    @staticmethod
    def from_dict(d: dict) -> "PiiDataEntry":
        attribute_keys = PiiDataEntry.__annotations__.keys()
        sd = {k: v for k, v in d.items() if k in attribute_keys}
        meta = {k: v for k, v in d.items() if k not in attribute_keys}
        if "metadata" in d and type(d["metadata"]) == dict:
            meta = meta | d["metadata"]
        for k in meta.keys():
            if k.startswith("__"):
                _ = meta.pop(k)
        sd["metadata"] = meta
        return PiiDataEntry(**sd)

    @staticmethod
    def from_spans(id: int | str, text: str, spans: list[LabelSpan]) -> "PiiDataEntry":
        words, word_spans = WordTokenizer().word_tokenize(text=text)
        return PiiDataEntry(
            id=id,
            text=text,
            tokens=words,
            token_spans=word_spans,
            bio_tags=spans_to_bio(
                label_spans=spans, token_spans=[Span(*x) for x in word_spans]
            ),
        )

    @staticmethod
    def example_pii() -> "PiiDataEntry":
        return PiiDataEntry.from_dict(
            {
                "text": "Rapport fra IT: en maskine med MAC-adresse loggede fra IP 10.0.0.42 og var knyttet til kontoret på Blegdamsvej 24, 2100 København Ø. Den ansvarlige medarbejder var registreret som Frederik Sørensen med e-mail frederik.sorensen@firma.dk og intern konto fsorensen.",
                "tokens": [
                    "Rapport",
                    "fra",
                    "IT",
                    ":",
                    "en",
                    "maskine",
                    "med",
                    "MAC",
                    "-",
                    "adresse",
                    "loggede",
                    "fra",
                    "IP",
                    "10",
                    ".",
                    "0",
                    ".",
                    "0",
                    ".",
                    "42",
                    "og",
                    "var",
                    "knyttet",
                    "til",
                    "kontoret",
                    "på",
                    "Blegdamsvej",
                    "24",
                    ",",
                    "2100",
                    "København",
                    "Ø",
                    ".",
                    "Den",
                    "ansvarlige",
                    "medarbejder",
                    "var",
                    "registreret",
                    "som",
                    "Frederik",
                    "Sørensen",
                    "med",
                    "e",
                    "-",
                    "mail",
                    "frederik",
                    ".",
                    "sorensen",
                    "@",
                    "firma",
                    ".",
                    "dk",
                    "og",
                    "intern",
                    "konto",
                    "fsorensen",
                    ".",
                ],
                "bio_tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-IP_ADDRESS",
                    "I-IP_ADDRESS",
                    "I-IP_ADDRESS",
                    "I-IP_ADDRESS",
                    "I-IP_ADDRESS",
                    "I-IP_ADDRESS",
                    "I-IP_ADDRESS",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-PRECISE_LOCATION",
                    "I-PRECISE_LOCATION",
                    "I-PRECISE_LOCATION",
                    "I-PRECISE_LOCATION",
                    "I-PRECISE_LOCATION",
                    "I-PRECISE_LOCATION",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-PERSON_NAME",
                    "I-PERSON_NAME",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-EMAIL_ADDRESS",
                    "I-EMAIL_ADDRESS",
                    "I-EMAIL_ADDRESS",
                    "I-EMAIL_ADDRESS",
                    "I-EMAIL_ADDRESS",
                    "I-EMAIL_ADDRESS",
                    "I-EMAIL_ADDRESS",
                    "O",
                    "O",
                    "O",
                    "B-USERNAME",
                    "O",
                ],
            }
        )


@dataclass
class PiiDataset(object):
    metadata: dict | None = None
    entries: list[PiiDataEntry] = field(default_factory=list)

    def __iter__(self):
        return self.entries.__iter__()

    def __len__(self):
        return len(self.entries)

    def labels(self):
        return {t[2:] for e in self.entries for t in e.bio_tags if t != "O"}

    def get_samples(self, samples: int):
        return random.sample(self.entries, min(samples, len(self)))

    @staticmethod
    def sample() -> "PiiDataset":
        return PiiDataset("sample_dataset", entries=[PiiDataEntry.example()])

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
        # return PiiDataset(**d)
        entries = []
        for i, e in enumerate(d["entries"]):
            if "id" not in e:
                e["id"] = i
            entries.append(PiiDataEntry.from_dict(e))
        d["entries"] = entries
        original_metadata = d.get("metadata") or dict()
        metadata = original_metadata | {
            k: v for k, v in d.items() if k not in PiiDataset.__annotations__.keys()
        }
        for k in metadata.keys():
            if k.startswith("__"):
                _ = metadata.pop(k)
        d["metadata"] = metadata
        return PiiDataset(**d)

    def to_json_str(self) -> str:
        return json.dumps(to_dict_recursive(self), ensure_ascii=False, indent=2)

    def to_file(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.write_text(self.to_json_str(), encoding="utf8")

    @staticmethod
    def from_span_dataset(
        name: str, dataset: list[tuple[int | str, str, list[LabelSpan]]]
    ):
        return PiiDataset(
            metadata={
                "dataset_name": name,
                "created": datetime.now().isoformat(),
            },
            entries=[
                PiiDataEntry.from_spans(id=id, text=text, spans=spans)
                for id, text, spans in dataset
            ],
        )

    @staticmethod
    def from_list_format_file(
        file: Path | str, dataset_name: str | None = None
    ) -> "PiiDataset":
        if isinstance(file, str):
            file = Path(file)
        assert file.exists()
        cnt: list[dict] = json.loads(file.read_text(encoding="utf8"))
        entries = []
        for entry in cnt:
            entry["tokens"] = entry.pop("words")
            entry["bio_tags"] = entry.pop("true_bio_tags")
            entry["id"] = entry.pop("document")
            entries.append(PiiDataEntry.from_dict(entry))
        return PiiDataset(
            entries=entries,
            metadata={
                "dataset_name": dataset_name if dataset_name else file.stem,
                "is_ground_truth": True,
                "file_path": str(file),
            },
        )
