import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from pii_eval.constants import LABEL_DESCRIPTIONS_DA
from pii_eval.data import PiiEntry, PiiModel
from pii_eval.span import LabelSpan, WordTokenizer, bio_to_spans

load_dotenv()


class NEREntityFormat(BaseModel):
    text: str = Field(..., description="Exact text of the entity")
    label: str = Field(..., description=f"Label of the entity")
    confidence: float = Field(..., description="Confidence score between 0 and 1")

    @property
    def example() -> str:
        return '{"text": "Danmarksvej nr. 4, 2400 Kbh NV", "label": "PRECISE_LOCATION"}'


class NERResponse(BaseModel):
    entities: list[NEREntityFormat]

    @staticmethod
    def example() -> str:
        return f'{{"entities": [{NEREntityFormat.example}, ...]}}'

    def to_bio(
        self, text: str, tokenizer=WordTokenizer(), min_confidence=0.0
    ) -> list[LabelSpan]:
        # Entities should be ordered; matching from the previous end disambiguates repeats.
        tokens = tokenizer(text)
        labels = ["O"] * len(tokens)
        search_start = 0

        for entity in self.entities:
            if not entity.text:
                continue
            if entity.confidence < min_confidence:
                continue

            label = entity.label
            if label == "O":
                continue
            if "-" in label:
                prefix, suffix = label.split("-", 1)
                if prefix in {"B", "I"}:
                    label = suffix

            start = text.find(entity.text, search_start)
            if start < 0:
                start = text.find(entity.text)
            if start < 0:
                logging.warning(
                    "Could not align entity text %r in input text.", entity.text
                )
                continue

            end = start + len(entity.text)
            token_indexes = [
                i
                for i, token in enumerate(tokens)
                if token.start < end and start < token.end
            ]
            if not token_indexes:
                logging.warning(
                    "Could not align entity %r to any token in input text.", entity.text
                )
                search_start = end
                continue

            for i in token_indexes:
                if labels[i] != "O":
                    continue
                previous_label = labels[i - 1] if i > 0 else "O"
                prefix = "I" if previous_label.endswith(f"-{label}") else "B"
                labels[i] = f"{prefix}-{label}"

            search_start = end

        return [
            LabelSpan(
                start=token.start,
                end=token.end,
                text=token.text,
                label=labels[i],
            )
            for i, token in enumerate(tokens)
        ]


class Generative(PiiModel):
    def __init__(
        self,
        model_tag: str = "qwen3.5-397b",
        base_url: str = "https://inference.alexandra.dk/v1/",
        api_key: str = os.environ.get("ALEX_API_KEY", None),
        label_descriptions: dict[str, str] = LABEL_DESCRIPTIONS_DA,
        name: str | None = None,
        min_confidence: float = 0.0,
    ):
        super().__init__(name=name if name else model_tag.split("/")[-1])
        if name is None:
            name = model_tag.split("/")[-1]
        self.name = name
        self.model_tag = model_tag
        self.base_url = base_url
        self.api_key = api_key
        self.min_confidence = min_confidence
        self.label_descriptions = label_descriptions
        self.model_json_schema = NERResponse.model_json_schema()
        self.preamble = (
            "You are an expert assistant that performs NER (named entity recognition) "
            "specifically for PII (person identifiable information) in contexts "
            "that can be in languages different from english (multilingual). "
        )
        self.client: OpenAI | None = None

    def __call__(self, entry: PiiEntry) -> PiiEntry:
        return PiiEntry(
            uid=entry.uid,
            text=entry.text,
            bio=self.predict_bio(entry.text, entry.bio),
            meta=entry.meta,
        )

    def predict_bio(self, text, bio):
        if self.client is None:
            self.start_client()
        relevant_descriptions = dict(self.label_descriptions)
        system_prompt = self.preamble + (
            "List of labels to use:\n"
            f"{list(relevant_descriptions.keys())}"
            "pick exactly one label for each PII substring/entity. "
            "Below are descriptions of each label in the relevant language:\n"
            f"{json.dumps(relevant_descriptions)}\n"
            "Use the descriptions only to understand the labels - "
            "for the actual output labelling use on the (upper-case) labels. "
            "Note that some texts may not contain any PII. "
            "However make sure to label anything and everything with the "
            "slightest chance of being PII. "
            "Rather label too many substrings (than too few) "
            "- aiming for high recall and low precision. "
            "Always label even doubtful substrings "
            "(just give them low confidence score)."
        )
        user_prompt = (
            "Label PII-entities in the following text:\n"
            f"'{text}'\n"
            "in the order they appear."
        )
        response = self.client.chat.completions.parse(
            model=self.model_tag,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=NERResponse,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        try:
            ner = NERResponse.model_validate(json.loads(content))
        except json.JSONDecodeError:
            logging.warning(f"Failed to load model output '{content}'.")
            ner = NERResponse(entities=[])

        bio = ner.to_bio(text, min_confidence=self.min_confidence)
        return bio

    def predict_spans(self, text):
        bio = WordTokenizer()(text)
        return bio_to_spans(self.predict_bio(text, bio))

    def start_client(self):
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
