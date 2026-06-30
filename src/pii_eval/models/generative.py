import json
import logging
import os

import bidict
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from pii_eval.constants import LABEL_DESCRIPTIONS_DA
from pii_eval.data import PiiEntry, PiiModel
from pii_eval.span import LabelSpan, WordTokenizer

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
        # The "entities are substrings of the text in the order they appear"
        raise NotImplementedError


class Generative(PiiModel):
    def __init__(
        self,
        model_tag: str = "qwen3.5-397b",
        base_url: str = "https://inference.alexandra.dk/v1/",
        api_key: str = os.environ.get("ALEX_API_KEY", None),
        label_descriptions: bidict[str, str] = LABEL_DESCRIPTIONS_DA,
        name: str | None = None,
        min_confidence: float = 0.0,
    ):
        if name is None:
            name = model_tag.split("/")[-1]
        self.model_tag = model_tag
        self.base_url = base_url
        self.api_key = api_key
        self.label_descriptions = label_descriptions
        self.model_json_schema = NERResponse.model_json_schema()
        self.preamble = (
            "You are a helpful assistant that performs NER (named entity recognition) "
            "specifically for PII (person identifiable information) in contexts "
            "that can be in languages different from english (multilingual). "
        )
        self.client: OpenAI | None = None

    def __call__(self, entry: PiiEntry) -> PiiEntry:
        if self.client is None:
            self.start_client()
        relevant_descriptions = self.label_descriptions
        system_prompt = self.preamble + (
            "List of labels to use:\n"
            f"{list(relevant_descriptions.keys())}"
            "pick exactly one label for each PII-sensitive substring/entity. "
            "Below are descriptions of each label in the relevant language:\n"
            f"{json.dumps(relevant_descriptions)}\n"
            "Use the descriptions only to understand the labels - "
            "for the actual output labelling use on the (upper-case) labels. "
            "Label anything and everything with the "
            "slightest chance of being PII-sensitive. "
            "Rather label too many substrings (than too few) "
            "- aiming for high recall and low precision. "
            "Always label even doubtful substrings "
            "(just give them low confidence score)."
        )
        user_prompt = (
            "Label all (possibly none) PII-entities in the following text:\n"
            f"'{entry.text}'\n"
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

        return ner.to_bio(entry.text)
