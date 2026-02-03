from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import litellm
import codetiming
import yaml

from dfm_anonymization.util import BlockStrDumper

language = "Danish"

DEFAULT_PROMPT = f"""Please generate {language} text material from contexts where PII
(personal identifiable information) may occur.
Wherever PII entities occur tag it using xml.
This is an example:

Det er forperson <person text="Jon Jensen"/>. Du kan skrive til ham på
<email text="jj@stofanet. com"/>. <phone text="+45)1234 5 678"/>"
er hans telefonnummer. Han blev født
<dob text="den 7. marts 1982" when="1982-03-07"/>.

Be creative with the text examples and categories.
Make sure:
- to vary the length of examples.
 the one given above, along with short onesto also include much longer ones.
The goal is to include as much different material and categories
as possible to provide data for a model that can capture all PII data
it may encounter.
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
    for _ in range(100):
        generate_from_prompt()
