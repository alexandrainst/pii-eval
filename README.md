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

# Dfm Anonymization

Anonymization in the context of [DFM](https://www.foundationmodels.dk/index.html)

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/dfm_anonymization/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/dfm_anonymization)
[![License](https://img.shields.io/github/license/alexandrainst/dfm_anonymization)](https://github.com/alexandrainst/dfm_anonymization/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/dfm_anonymization)](https://github.com/alexandrainst/dfm_anonymization/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/dfm_anonymization/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Jacob Harder (<jacob.harder@alexandra.dk>)

## Setup

### Installation

1. Run `make install`, which sets up a virtual environment and all Python dependencies
   therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Example usage

```bash
# Anonymize inline text
dfm-anonymization anonymize \
  "Navnet er Anders Sand. Og 21324354 er telefonnummeret." \
  --whitelist "Anders Sand"

# Anonymize piped input
echo "Eksempel telefonnummer: 12 23 34 45" | dfm-anonymization anonymize

# Anonymize a file
dfm-anonymization anonymize ./text_file.txt > ./text_file_anonymized.txt

# Generate evaluations
dfm-anonymization make-evaluations

# Start the API
dfm-anonymization start-api
# Or via
uvicorn dfm_anonymization.service:app --host 0.0.0.0 --port 8000
# then anonymize text via HTTP
curl -X POST "http://localhost:8000/anonymize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Navnet er Anders Sand. Og 21324354 er telefonnummeret.","whitelist":["Anders Sand"]}'

# Get help
dfm-anonymization --help
dfm-anonymization anonymize --help
```

### Performance

Current (limited) test results:
| Hardware | Speed |
| --- | --- |
| CPU (x86_64, Intel(R) Core(TM) Ultra 7 258V) | ~1.5 sentences / second |
| GPU (RTX 4090) | ~170 sentences / second |
(1 sentence ≃ 250 tokens)
