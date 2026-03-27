import os
from datetime import datetime
from pathlib import Path

import codetiming
import openai
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from dfm_anonymization.constants import LABEL_DESCRIPTIONS_DA
from dfm_anonymization.util import BlockStrDumper, Span


class GenItem(BaseModel):
    text: str = Field(...)
    spans: list[Span] = Field(...)


class GenOutput(BaseModel):
    items: list[GenItem] = Field(...)


language = "Danish"

SYSTEM_PROMPT = (
    "You are an expert in PII (personal identifiable information) "
    "and synthetic data generation."
)

USER_PROMPT = f"""
Your task is to generate {language} text material from contexts where PII may occur.
Use the following categories of PII
(label names in upper case and their descriptions following in lower case):

{LABEL_DESCRIPTIONS_DA}

Information should be labelled if and only if it endangers anonymity.
This includes information which can be identifying due to its context.

Be creative with the texts and make sure to
- include all the above categories
- vary the length of examples (loosely between 25 to 500 words)
The goal is to include as much different material and categories
as possible to provide data for a model that can capture all PII data
it may encounter.
- vary the density of PII information (e.g. also include a few examples with no PII data)
- the examples are similar to how PII would appear in actual {language}
documents/texts.

For the output you use the following format (model json schema):

{GenOutput.model_json_schema()}

containing a list of items, each consisting of a text and spans
with (start and end) indices and labels of its PII entities.
"""


def generate_from_prompt(
    count: int,
    model: str = "gpt-5-mini",
    base_url: str = "https://api.openai.com/v1",
    api_key: str = os.getenv("OPENAI_API_KEY"),
    output_dir: Path | str = Path("data/raw/generated/prompt_alpha"),
):
    load_dotenv()
    if type(output_dir) == str:
        output_dir = Path(output_dir).resolve()
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    for _ in tqdm(range(count), dynamic_ncols=True):
        try:
            with codetiming.Timer("get completion"):
                response = client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                    ],
                    response_format=GenOutput,
                )
            json_str = response.choices[0].message.content
            gen_output = GenOutput.model_validate_json(json_str)
        except Exception as e:
            print(e)
            continue

        output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().isoformat()
        output_path = output_dir / f"{timestamp}.yaml"
        output_path.write_text(
            yaml.dump(
                gen_output.model_dump(),
                Dumper=BlockStrDumper,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf8",
        )


if __name__ == "__main__":
    generate_from_prompt(10)
