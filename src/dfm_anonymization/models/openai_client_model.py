import json
import logging
import os
from typing import Set

from bidict import bidict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from dfm_anonymization.constants import LABEL_DESCRIPTIONS_DA
from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.util import Span, spans_to_bio


class OpenAIClientModel(PiiModel):
    def __init__(
        self,
        model_tag: str,
        base_url: str = "http://localhost:8000",
        api_key: str = "not_needed",
        label_descriptions: Set[str] | bidict[str, str] = LABEL_DESCRIPTIONS_DA,
    ):
        super().__init__(name=model_tag.split("/")[-1])
        self.model_tag = model_tag
        self.base_url = base_url
        self.api_key = api_key
        self.label_descriptions = label_descriptions
        self.model_json_schema = NERResponse.model_json_schema()
        self.preamble = (
            "You are a helpful assistant that performs NER (named entity recognition) "
            "specifically for PII (person identifiable information) in contexts "
            "that can be in languages different from english (multilingual). "
            "Answer so that the entire output always deserializes to the following "
            f"model JSON schema:\n{self.model_json_schema}\n"
            "Never output newlines, multiple spaces or explanatory text. "
            f"Here's an example output:\n{NERResponse.example}\n"
        )
        self.client: OpenAI | None = None

    def __call__(
        self, entry: PiiDataEntry, label_whitelist: Set[str] | None = None
    ) -> list[str]:
        if self.client is None:
            self.start_client()
        relevant_descriptions: dict = {
            k: v
            for k, v in self.label_descriptions.items()
            if label_whitelist is None or k in label_whitelist
        }
        system_prompt = self.preamble + (
            "List of labels to use:\n"
            f"{list(relevant_descriptions.keys())}"
            "pick one for each PII-sensitive substring/entity. "
            "Below are descriptions of each label in the relevant language:\n"
            f"{json.dumps(relevant_descriptions)}\n"
            "Use only the descriptions to understand the labels - "
            "for the actual output labelling use on the (upper-case) labels. "
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
        )
        content = response.choices[0].message.content
        try:
            ner = NERResponse.model_validate(json.loads(content))
        except json.JSONDecodeError:
            logging.warning(f"Failed to load model output '{content}'.")
            ner = NERResponse(entities=[])

        return ner.to_bio(entry.text, entry.get_token_spans())

    def start_client(self):
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)


class NEREntityFormat(BaseModel):
    text: str = Field(..., description="Exact text of the entity")
    label: str = Field(..., description=f"Label of the entity")
    confidence: float = Field(..., description="Confidence score between 0 and 1")

    @property
    def example() -> str:
        return '{"text": "Danmarksvej nr. 4, 2400 Kbh NV", "label": "PRECISE_LOCATION"}'


class NERResponse(BaseModel):
    entities: list[NEREntityFormat]

    @property
    def example() -> str:
        return f'{{"entities": [{NEREntityFormat.example}, ...]}}'

    def to_bio(self, text: str, token_spans: list[tuple[int, int]]) -> list[str]:
        spans = []
        cursor = 0
        for entity in self.entities:
            start = text.find(entity.text, cursor)
            if start >= 0:
                cursor = start
                end = start + len(entity.text)
                spans.append(Span(start=start, end=end, label=entity.label))
            else:
                logging.warning(
                    f"Failed to find model output substring "
                    f"'{entity.text}' in text '{text[cursor:]}'."
                )
        return spans_to_bio(token_spans=token_spans, spans=spans)


if __name__ == "__main__":
    load_dotenv()

    model = OpenAIClientModel(
        model_tag="google/gemma-3n-e4b-it",  # "meta-llama/llama-3.2-3b-instruct:free",  # "qwen/qwen3.5-27b",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    sample = PiiDataEntry.example_pii()
    x = model(sample)

    print(sample)
    print(x)
    pass
