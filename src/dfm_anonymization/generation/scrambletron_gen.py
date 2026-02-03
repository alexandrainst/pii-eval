import os
from pathlib import Path
from data_generation.generate import SampleGenerator
from data_generation.languages import DA
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

gen = SampleGenerator(
    model="openai/gpt-5-mini",
    num_samples=5,
    base_num_tokens=None,
    tokens_per_sample=None,
    language=DA,
    templates_dir=Path.cwd(),
    base_prompt_file=Path("config/base_prompt.jinja"),
    temperature=1,
    # base_url="https://inference.projects.alexandrainst.dk/v1/chat/completions",
    base_url="https://api.openai.com/v1",
    api_key=api_key,
)

gen.generate()

pass
