# DFM Anonymisation

The aim of this project is to evaluate and pipeline methods for PII
(personally identifiable information) detection and removal in plain text.
So far the main focus has been on the *Danish* language, though much should
generalize to other scandinavian/germanic languages.

## Quick start

See README.md

## Data and ground truth

Public Danish PII datasets does not seem to be in abundance.
The ground truth so far used for evaluation is:

- [DaNE](https://huggingface.co/datasets/alexandrainst/dane)
  - Only NER tags ORG, LOC, and PER are used,
  other categories are deemed not relevant in PII context
- [Dansk-NER](https://huggingface.co/datasets/chcaa/dansk-ner)
  - Only NER tag PERSON is used,
  other categories are deemed not relevant in PII context
- Generated data using the [Presidio](https://microsoft.github.io/presidio/)
library with the categories (160-170 sentences for each category):
  - ADDRESS
  - CARD EXPIRY
  - CARD NUMBER
  - CPR NUMBER
  - CVV NUMBER
  - DATE OF BIRTH
  - DRIVERS LICENSE
  - EMAIL
  - ORGANIZATION
  - PERSON
  - PHONE NUMBER
  - WEBSITE

## Models

As of now we still have too little annotated data for actual (post) training
of NER models specifically for PII. Therefore we so far only use off-the-shelf
models like GliNER models or NER models with some overlap in categories with
PII.

These are the models we looked at so far:

- [scandiner](https://huggingface.co/saattrupdan/nbailab-base-ner-scandi)
  - Using categories ORG, LOC, PER
- presidio
  - the standard presidio NER model for Danish. Based on
  the spacy model [*da_core_news_trf*](https://spacy.io/models/da#da_core_news_trf)
- [gliner-PII](https://huggingface.co/nvidia/gliner-PII)
- [gliner-multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1)

The model used by default by the CLI tool (`dfm-anonymization`) is an
ensemble model based on gliner-pii and presidio.
This is chosen since these two together are close to best in every category,
while introducing more models would hurt performance.

##

## Evaluation results

We evaluate per category and per model whenever both dataset and model include
this category, then average the scores for each dataset (and then each category).
Finally we rank based on the f5 score, weighing recall higher than precision.

| model | precision | f1 | f5 | recall |
| --- | --- | --- | ---| --- |
| ensemble | 0.965112 | 0.945467 | 0.932161 | 0.931222 |
| scandiner | 0.960480 | 0.934935 | 0.915107 | 0.913586 |
| gliner-pii | 0.876750 | 0.851719 | 0.854775 | 0.856366 |
| gliner_multi-v2.1 | 0.733916 | 0.778923 | 0.820998 | 0.722963 |
| presidio | 0.967630 | 0.818236 | 0.769630 | 0.766570 |

### Full results for each category:

| label | model | precision | f1 | f5 | recall |
| --- | --- | --- | --- | --- | --- |
| card     | gliner-pii                   |  1.000000 | 0.730123 | 0.584511 | 0.574956 |
| time       | gliner-pii                   |  0.936940 | 0.967381 | 0.997408 | 1.000000 |
|                 | gliner_multi            |  0.959983 | 0.977574 | 0.994465 | 0.995902 |
|                 | presidio                     |  0.984375 | 0.330753 | 0.205657 | 0.199378 |
| drivers license | gliner-pii                   |  1.000000 | 1.000000 | 1.000000 | 1.000000 |
|                 | gliner_multi            |  0.883041 | 0.909639 | 0.935653 | 0.937888 |
| email           | gliner-pii                   |  0.997854 | 0.983087 | 0.969838 | 0.968750 |
|                 | presidio                     |  1.000000 | 0.965517 | 0.935733 | 0.933333 |
|                 | gliner_multi            |  0.795699 | 0.258290 | 0.159100 | 0.154167 |
| iban            | gliner-pii                   |  0.987730 | 0.975758 | 0.964961 | 0.964072 |
|                 | gliner_multi            |  0.549587 | 0.650367 | 0.782884 | 0.796407 |
| loc        | presidio                     |  0.971716 | 0.944454 | 0.921885 | 0.920105 |
|                 | scandiner      |  0.983142 | 0.919632 | 0.871915 | 0.868313 |
|                 | gliner_multi            |  0.857555 | 0.812221 | 0.777663 | 0.775044 |
|                 | gliner-pii                   |  0.799862 | 0.691106 | 0.636982 | 0.633698 |
| org   | scandiner      |  0.983658 | 0.963033 | 0.946002 | 0.944662 |
|                 | presidio                     |  0.990338 | 0.957231 | 0.930327 | 0.928224 |
|                 | gliner_multi            |  0.705424 | 0.761888 | 0.882310 | 0.898388 |
|                 | gliner-pii                   |  0.695549 | 0.740441 | 0.853925 | 0.869835 |
| per          | scandiner      |  0.929921 | 0.926404 | 0.923304 | 0.923051 |
|                 | presidio                     |  0.906235 | 0.913072 | 0.920433 | 0.921096 |
|                 | gliner_multi            |  0.753707 | 0.805625 | 0.875209 | 0.882343 |
|                 | gliner-pii                   |  0.747345 | 0.777887 | 0.819428 | 0.823601 |
| phone           | gliner_multi            |  0.979920 | 0.989858 | 0.999212 | 1.000000 |
|                 | gliner-pii                   |  1.000000 | 0.993814 | 0.988172 | 0.987705 |
|                 | presidio                     |  1.000000 | 0.786070 | 0.656440 | 0.647541 |
| ssn             | gliner-pii                   |  0.935673 | 0.958084 | 0.979746 | 0.981595 |
|                 | gliner_multi            |  0.493450 | 0.576531 | 0.682621 | 0.693252 |
| url             | presidio                     |  1.000000 | 0.863158 | 0.766355 | 0.759259 |
|                 | gliner-pii                   |  1.000000 | 0.855124 | 0.754256 | 0.746914 |
