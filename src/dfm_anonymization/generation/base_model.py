import os
from datetime import datetime
from math import prod
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dfm_anonymization.util.util import BlockStrDumper


def generate_from_base_model():
    # prompt = Path("assets/prompts/base/no_seed.txt").read_text()
    seed_prefix = Path("assets/prompts/base/seed_prefix.txt").read_text()
    seed = Path("assets/annotations/marianne_nilsen.txt").read_text()
    prompt = "\n\n".join([seed_prefix, "Tekst 1", seed, "Tekst 2"])

    max_length = 10000
    # model_tag = "danish-foundation-models/dfm-decoder-open-v0-7b-pt"
    model_tag = "google/gemma-3-12b-pt"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit
        bnb_4bit_quant_type="nf4",  # nf4 > fp4 for quality typically
        # bnb_4bit_use_double_quant=True,         # double quantization reduces VRAM
        bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 on older GPUs
    )

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_tag, hf_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_tag,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        # low_cpu_mem_usage=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = prod(inputs.input_ids.shape)
    outputs = model.generate(
        **inputs,
        max_length=max_length + input_len,
        do_sample=True,
    )
    output = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(output)
    timestamp = datetime.now().isoformat()
    output_path = Path(f"data/raw/generated/dfm-anonymization-base/{timestamp}.yaml")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(
        yaml.dump(
            {"prompt": prompt, "content": output},
            Dumper=BlockStrDumper,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf8",
    )
    pass


if __name__ == "__main__":
    generate_from_base_model()
