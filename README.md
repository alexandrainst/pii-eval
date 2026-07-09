<!-- This disables the "First line in file should be a top level heading" rule -->
<!-- markdownlint-disable MD041 -->
<a href="https://github.com/alexandrainst/dfm_anonymization">
<img
 src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/alexandra/alexandra-logo.jpeg"
 width="239"
 height="175"
 align="right"
 alt="Alexandra Institute Logo"
/>
</a>

# pii-eval

Evaluation of PII (personal identifiable information) detection and redaction.
Currently primarily focused on the Danish language.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/dfm_anonymization/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/dfm_anonymization)
[![License](https://img.shields.io/github/license/alexandrainst/dfm_anonymization)](https://github.com/alexandrainst/dfm_anonymization/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/dfm_anonymization)](https://github.com/alexandrainst/dfm_anonymization/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/dfm_anonymization/blob/main/CODE_OF_CONDUCT.md)

Developer:
- Jacob Harder (<jacob.harder@alexandra.dk>)

## Setup

1. Run `make install`, which sets up a virtual environment and all Python dependencies
   therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. If you want to use generative models via an API, you need to
   `cp pii_eval_config.example.yaml pii_eval_config.yaml` and fill out the
   necessary details in this file (api keys etc. and a URL to an OpenAI-compatible
   API).

## Usage

### Python package

```python
from pii_eval.models.ensemble import Ensemble
from pii_eval.redact import redact

sample_text = "Navnet er Anders Sand. Og 21324354 er telefonnummeret."

print(redact(sample_text))
# Output: Navnet er <PERSON_NAME>. Og <PHONE_NUMBER> er telefonnummeret.

ensemble_model = Ensemble.default()

spans = ensemble_model.predict_spans(sample_text)

print(spans)
# Output:
# [
#   {
#     "start": 10,
#     "end": 21,
#     "text": "Anders Sand",
#     "label": "PERSON_NAME"
#   },
#   {
#     "start": 26,
#     "end": 34,
#     "text": "21324354",
#     "label": "PHONE_NUMBER"
#   }
# ]
```

### CLI

```bash
# Redact text
pii-eval redact "Navnet er Anders Sand. Og 21324354 er telefonnummeret."
> Navnet er <PERSON_NAME>. Og <PHONE_NUMBER> er telefonnummeret.

# Or get the entities in JSON format:
pii-eval identify "Navnet er Anders Sand. Og 21324354 er telefonnummeret."
[
  {
    "start": 10,
    "end": 21,
    "text": "Anders Sand",
    "label": "PERSON_NAME"
  },
  {
    "start": 26,
    "end": 34,
    "text": "21324354",
    "label": "PHONE_NUMBER"
  }
]

# Run evaluations
pii-eval run-evaluation

# CLI docs
pii-eval --help
pii-eval redact --help
```
