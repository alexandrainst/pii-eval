from enum import Enum
import json
import logging
from typing import List, Set

from bidict import bidict

from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from dfm_anonymization.constants import LABEL_DESCRIPTIONS_DA
from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel
from dfm_anonymization.util import Span, spans_to_bio


# logging.getLogger().setLevel(level=logging.ERROR)
#
# import os
#
# # Set logging level before importing/initializing vLLM
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"


class GenModel(PiiModel):
    def __init__(
        self,
        hf_tag: str,
        label_descriptions: Set[str] | bidict[str, str] = LABEL_DESCRIPTIONS_DA,
    ):
        super().__init__(name=hf_tag.split("/")[-1])
        self.label_descriptions = label_descriptions
        self.model_json_schema = NERResponseFormat.model_json_schema()
        self.preamble = (
            "You are a helpful assistant that performs NER (named entity recognition) "
            "specifically for PII (person identifiable information) in contexts "
            "that can be in languages different from english (multilingual). "
            "Answer so that the entire output always deserializes to the following "
            f"model JSON schema:\n{self.model_json_schema}\n"
            "Never output newlines, multiple spaces or explanatory text. "
            f"Here's an example output:\n{NERResponseFormat.example}\n"
        )
        self.llm = LLM(model=hf_tag, gpu_memory_utilization=0.9)

    def __call__(
        self, entry: PiiDataEntry, label_whitelist: Set[str] | None = None
    ) -> List[str]:
        relevant_descriptions: dict = {
            k: v
            for k, v in self.label_descriptions.items()
            if label_whitelist is None or k in label_whitelist
        }
        prompt = (
            "List of labels to use:\n"
            f"{list(relevant_descriptions.keys())}"
            "pick one for each PII-sensitive substring/entity. "
            "Below are descriptions of each label in the relevant language:\n"
            f"{json.dumps(relevant_descriptions)}\n"
            "Only use the descriptions to understand the labels - "
            "for the actual output labelling use on the (upper-case) labels. "
            "Label all (possibly none) PII-entities in the following text:\n"
            f"'{entry.text}'\n"
            "in the order they appear."
        )
        tokenizer = self.llm.get_tokenizer()
        newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        sampling_params = SamplingParams(
            temperature=0,
            structured_outputs=StructuredOutputsParams(
                json=self.model_json_schema,
                disable_any_whitespace=True,
                disable_fallback=True,

            ),
            max_tokens=300,
            logit_bias={newline_token_id: -1000},
        )
        prompt = "\n".join([self.preamble, prompt])
        output = self.llm.generate(
            prompts=prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        content = output[0].outputs[0].text
        try:
            ner = NERResponseFormat.model_validate(json.loads(content))
        except json.JSONDecodeError:
            logging.warning(f"Failed to load model output '{content}'.")
            ner = NERResponseFormat(entities=[])
        spans = []
        cursor = 0
        for entity in ner.entities:
            start = entry.text.find(entity.substring, cursor)
            if start >= 0:
                cursor = start
                end = start + len(entity.substring)
                spans.append(Span(start=start, end=end, label=entity.label))
            else:
                logging.warning(
                    f"Failed to find model output substring "
                    f"'{entity.substring}' in text '{entry.text}'."
                )
        bio_tags = spans_to_bio(token_spans=entry.get_token_spans(), spans=spans)
        return bio_tags

class PIILabel(Enum):


class NEREntityFormat(BaseModel):
    substring: str
    label: str

    @property
    def example() -> str:
        return '{"substring": "..some exact substring of text.", "label": "LABEL"}'


class NERResponseFormat(BaseModel):
    entities: list[NEREntityFormat]

    @property
    def example() -> str:
        return f'{{"entities": [{NEREntityFormat.example}, ...]}}'


if __name__ == "__main__":
    hf_tag = "Qwen/Qwen3-0.6B"

    model = GenModel(
        hf_tag=hf_tag,
        label_descriptions=bidict({"n": "navn", "m": "verbum (udsagnsord)"}),
    )

    sample = PiiDataEntry.sample()
    x = model(sample)

    print(sample)
    print(x)
    pass
