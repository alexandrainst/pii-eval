import datetime
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from faker.providers import BaseProvider
from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import PresidioSentenceFaker
from tqdm import tqdm

sentence_templates = [
    "My name is {{name}}",
    "Please send it to {{address}}",
    "I just moved to {{city}} from {{country}}",
]


sentence_faker = PresidioSentenceFaker(
    "en_US", lower_case_ratio=0.05, sentence_templates=sentence_templates
)
fake_sentence_results = sentence_faker.generate_new_fake_sentences(10)

# Print the spans of the first sample
print(fake_sentence_results[0].masked)
print(fake_sentence_results[0].spans)

number_of_samples = 1500
lower_case_ratio = 0.05
locale = "en"
cur_time = datetime.date.today().strftime("%B_%d_%Y")

data_fp = Path("data/raw/presidio")
output_file = data_fp / f"generated_size_{number_of_samples}_date_{cur_time}.json"
output_conll = data_fp / f"generated_size_{number_of_samples}_date_{cur_time}.tsv"

sentence_faker = PresidioSentenceFaker("en_US", lower_case_ratio=0.05)

print(pd.DataFrame(sentence_faker._sentence_faker.records).head())


class MarsIdProvider(BaseProvider):
    def mars_id(self):
        # Generate a random row number between 1 and 50
        row = random.randint(1, 50)
        # Generate a random letter for the seat location from A-K
        location = random.choice("ABCDEFGHIJK")
        # Return the seat in the format "row-letter" (e.g., "25A")
        return f"{row}{location}"


sentence_faker.add_provider(MarsIdProvider)
# Now a new `mars_id` entity can be generated if a template has `mars_id` in it.
provider_aliases = PresidioSentenceFaker.PROVIDER_ALIASES

fake_records = sentence_faker.generate_new_fake_sentences(num_samples=number_of_samples)
print(fake_records[0])

count_per_template_id = Counter([sample.template_id for sample in fake_records])

print(f"Total: {sum(count_per_template_id.values())}")
avg_rec_per_template = np.mean(list(count_per_template_id.values()))
med_rec_per_template = np.median(list(count_per_template_id.values()))
print(f"Avg # of records per template: {avg_rec_per_template}")
print(f"Median # of records per template: {med_rec_per_template}")
print(f"Std: {np.std(list(count_per_template_id.values()))}")

count_per_entity: Counter = Counter()
for record in fake_records:
    count_per_entity.update(Counter([span.entity_type for span in record.spans]))

print(count_per_entity)

for record in fake_records[:10]:
    print(record)

output_file.parent.mkdir(exist_ok=True, parents=True)
InputSample.to_json(dataset=fake_records, output_file=output_file)

PRESIDIO_SPACY_ENTITIES = dict(
    PERSON="PERSON",
    LOCATION="LOC",
    GPE="GPE",
    ORGANIZATION="ORG",
    DATE_TIME="DATE",
    NRP="NORP",
)


def to_conll(
    input_sample: InputSample, translate_tags: bool, tokenizer: str = "en_core_web_sm"
) -> List[Dict[str, Any]]:
    """
    Turns a list of InputSample objects to a dictionary
    containing text, pos, tag, template_id and label.
    :param translate_tags: Whether to translate tags using the PRESIDIO_SPACY_ENTITIES dictionary
    :param tokenizer: The name of the spaCy model to use for tokenization
    :return: Dict
    """

    conll = []

    if len(input_sample.tokens) == 0:
        input_sample.tokens, input_sample.tags, _ = input_sample.get_tags(
            model_version=tokenizer
        )

    for i, token in enumerate(input_sample.tokens):
        if translate_tags:
            label = input_sample.translate_tag(
                input_sample.tags[i], PRESIDIO_SPACY_ENTITIES, ignore_unknown=True
            )
        else:
            label = input_sample.tags[i]
        conll.append(
            {
                "text": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "template_id": input_sample.template_id,
                "label": label,
            },
        )

    return conll


def create_conll_dataset(
    dataset: List["InputSample"],
    translate_tags=False,
    to_bio=True,
    tokenizer: str = "en_core_web_sm",
) -> pd.DataFrame:
    if len(dataset) <= 1:
        raise ValueError("Dataset should contain multiple records")

    conlls = []
    i = 0
    for sample in tqdm(dataset):
        if to_bio:
            sample.biluo_to_bio()
        conll = to_conll(sample, translate_tags=translate_tags, tokenizer=tokenizer)
        for token in conll:
            token["sentence"] = i
            conlls.append(token)
        i += 1

    return pd.DataFrame(conlls)


conll = create_conll_dataset(dataset=fake_records)
conll.head(10)

output_conll.parent.mkdir(exist_ok=True, parents=True)
conll.to_csv(output_conll, sep="\t")
print(f"CoNLL2003 dataset structure output location: {output_conll}")
