from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import litellm
import codetiming
from tqdm import tqdm
import yaml

from dfm_anonymization.util import BlockStrDumper

language = "Danish"

SHORT_EXAMPLE = """Det er forperson <person text="Jon Jensen"/>.
Du kan skrive til ham på <EMAIL_ADDRESS text="jj@stofanet.com"/>.
<PHONE_NUMBER text="(+45) 12 34 56 78"/> er hans telefonnummer.
Han blev født <OTHER_PII text="den 7. marts 1982"/>.
"""

LABELS = """
BANK_ACCOUNT
DRIVERS_LICENSE
EMAIL_ADDRESS
IP_ADDRESS
OTHER_PII
PASSPORT
PCI (payment card information)
PERSON_NAME
PHONE_NUMBER
PRECISE_LOCATION
PRIVATE_URL
SSN (social security number)
USERNAME
VEHICLE_ID
"""

DEFAULT_PROMPT = f"""Your task is to generate {language} text material
from contexts where PII (personal identifiable information) may occur.
We use the following categories of PII:
{LABELS}

Only information which endangers anonymity can fall into any of these categories.
This includes however also if other context can reveal identity.
Wherever PII entities occur it should be tagged with the scheme
<CATEGORY text="string to tag"/>.

Here is a short example:
'{SHORT_EXAMPLE.strip()}'

Be creative with the text examples and categories.
Make sure:
- include all the above categories
- to vary the length of examples (can be x10 times longer than the one above or more).
The goal is to include as much different material and categories
as possible to provide data for a model that can capture all PII data
it may encounter.
- to vary the density of PII information (e.g. also include a few examples with no PII data)
- the examples are similar to how PII would appear in actual {language}
documents/texts.
- the response only contain the examples and nothing else, that is,
no numeration, no suggestions, no explanations of the categories.
Not even introductions like "short example:" or "long example: ".
- the xml tags only contain exactly the PII-sensitive text and nothing else.
"""


def generate_from_prompt(
    prompt: str = DEFAULT_PROMPT,
    model: str = "gpt-5-mini",
    base_url: str = None,
    api_key: str = os.getenv("LLM_API_KEY"),
    output_dir: Path | str = Path("data/raw/generated/dfm-anonymization-gen"),
):
    with codetiming.Timer("get completion"):
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            base_url=base_url,
            api_key=api_key,
        )
    if type(output_dir) == str:
        output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    content = response.choices[0].message.content
    print(content)
    timestamp = datetime.now().isoformat()
    output_path = output_dir / f"{timestamp}.yaml"
    output_path.write_text(
        yaml.dump(
            {"prompt": prompt.strip(), "response_content": content.strip()},
            Dumper=BlockStrDumper,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    load_dotenv()
    for _ in tqdm(range(100), dynamic_ncols=True):
        generate_from_prompt()
