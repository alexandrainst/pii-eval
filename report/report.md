## OCR and redaction with open models

### Introduction

This is an investigation into the performance of current SOTA open (weight) models for
OCR (optical character recognition) and PII-redaction
(removal of personally identifiable information from text) specifically
for the dataset *dimmi*.

### Conclusion

Based on the evaluation results, if perhaps not surprising it is clear that the
large multi-modal language model Qwen3.5-397B outperforms smaller open model on all
metrics (except speed). Therefore, for best quality we recommend to use this model
for both tasks.
However if we take speed and price into account there may be something to save
by using e.g. Tesseract for large amounts of data for OCR.
As for redaction we would argue that the ~20% precision (for >95% recall) of smaller open models simply
is not good enough, and a larger model like Qwen3.5-397B is necessary
(20% precision will most likely harm the data quality too much).

### The *dimmi* dataset

The dimmi dataset consists of (mock) contracts, CVs, diplomas and passports.
The raw data consist of the following properties:

| column     | type          | meaning |
| ---        | ---           | --- |
| `document_id` | string        | stable identifier |
| `filename` | string        | relative path |
| `text`     | string        | reference transcription used as ground truth by the OCR evaluator |
| `pii`      | JSON string   | list of `{category, start, end, text}` PII spans, with `start`/`end` as Python-style character offsets into the `text` column  - ground truth for the PII redaction evaluation|

#### Aggregate statistics

| dataset                                              | documents | format | `text` chars (min / median / max) | total PII spans | avg spans / doc |
| ---                                                  | ---:      | ---    | ---:                              | ---:            | ---:            |
| [dimmi-contracts](../data/raw/dimmi-contracts/)      | 53        | PDF    | 1 262 / 2 491 / 6 991             | 678             | 12.8            |
| [dimmi-cvs](../data/raw/dimmi-cvs/)                  | 60        | PDF    | 290 / 777 / 1 166                 | 424             | 7.1             |
| [dimmi-diplomas](../data/raw/dimmi-diplomas/)        | 20        | PDF    | 3 092 / 9 969 / 22 179            | 238             | 11.9            |
| [dimmi-passport](../data/raw/dimmi-passport/)        | 24        | JPG    | 229 / 672 / 1 079                 | 135             | 5.6             |

### Evaluation of OCR

#### Overall performance

| model | mean_cer | mean_wer | mean_ms_per_word |
| --- | --- | --- | --- |
| qwen3.5-397b     | **0.12** | **0.34** | 104.8 |
| PaddleOCR-VL-1.5 | 0.34 | 0.43 | 76.6 |
| Tesseract        | 0.34 | 0.66 | **16.6** |

CPU adds ~20x computation time for locally run models.
For long documents this means ~20m with PaddleOCR vs. ~4m with Tesseract.
For Qwen3.5-397B, computation is on remote hardware, so computation time will be (effectively) the same on CPU.

#### Method

All documents (PDFs) and images were converted using python libraries `fitz, PIL.Image`
using `dpi=150`.

##### Model-specific details

- PaddleOCR-VL-1.5 was run using python library `paddleocr.PaddleOCRVL.predict` pipeline.
simply joining each "block_content" with newlines.
- Tesseract was run using the pytesseract python library `pytesseract`'s method `image_to_string`.
- Generative models was run using the following system and user prompts:
  - System prompt: "*You are an expert in extracting text from documents and images. You always arrange the extracted output text in a sensible fashion, reflecting the structure of the text as it was found in the source, but not sacrificing having a logical final output structure. You always only output the actual text found in the sources, never your own replies to the prompt or markdown formatting etc.*"
  - User prompt: "*Please extract all text from the following content: \<images(base64)\>*"

### Evaluation of Redaction

#### Best feasible models (>95% recall)

| model                         |   precision  |   recall |   seconds_per_document |
|:------------------------------|------------: |---------:|-----------------:|
| qwen3.5-397b                  | **0.726496** | 0.952915 |       37.370 |
| eu-pii-safeguard_biased_-11.5 |   0.207728   | 0.953279 |        0.063 |
| gliner-pii_th_0.0002          |   0.106378   | 0.950183 |        0.214 |

Again CPU adds ~20x to time for locally hosted models (Qwen3.5-397B is invariant).

#### Method

All models was asked to classify tokens, or identify substrings in the texts labelling them
in one of the following categories and descriptions (in Danish):

- BANK_ACCOUNT: "bankkontonummer"
- DRIVERS_LICENSE: "kørekortnummer"
- EMAIL_ADDRESS: "email-adresse"
- IP_ADDRESS: "IP-adresse"
- PASSPORT: "pasnummer"
- PCI: "betalingskortoplysninger, kortnummer, udløb, CVV"
- PERSON_NAME: "personnavn"
- PHONE_NUMBER: "telefonnummer (alle landekoder)"
- PRECISE_LOCATION: "lokation - gadenavn, husnummer, postnummer, koordinater, etc."
- SALARY: "løn - fx. løntrin, timeløn eller månedslån beløb, etc."
- SSN: "CPR-nummer"
- USERNAME: "brugernavn (fx. på online platforme)"
- VEHICLE_ID: "registreringsnummer / nummerplade"

The tokens and substrings were then converted to a unified word-tokenization format
using the `spacy` library with the danish word-tokenization model *da_core_news_sm*:
`nlp = spacy.load("da_core_news_sm"); nlp(text)`.
Further, the labels were merged into a single label `PII`, and the scores were calculated
based on true/false positive/negative counts of the `PII` and `O` labels.

#### Biasing and thresholds

- The model *eu-pii-safeguard* (huggingface tag: tabularisai/eu-pii-safeguard) was biased
using logit O-biasing by -11.5. This means that 11.5 was subtracted the value of the
logit for the 'O' label before `argmax` was applied to obtain the prediction of each
token. In practice this means that the IDs of all other labels (all of which predicts
some PII label or other) have a higher chance of being predicted - resulting in higher
recall and lower precision.

- The *gliner-pii* model (huggingface tag: nvidia/gliner-PII) uses the `gliner` library
(https://urchade.github.io/GLiNER/), which features a built-in threshold parameter
for tuning precision/recall. Lower values result in higher recall and lower precision.
Default is 0.5, and the threshold model with higher than 95% recall on the datasets was
found to be 0.0002 - which was used to obtain the above results.

- *Qwen3.5-397B* was run using the following prompts:
```
system_prompt: You are a helpful assistant that performs NER (named entity recognition) specifically for PII (person identifiable information) in contexts that can be in languages different from english (multilingual). Answer so that the entire output always deserializes to the following model JSON schema:
{'$defs': {'NEREntityFormat': {'properties': {'text': {'description': 'Exact text of the entity', 'title': 'Text', 'type': 'string'}, 'label': {'description': 'Label of the entity', 'title': 'Label', 'type': 'string'}, 'confidence': {'description': 'Confidence score between 0 and 1', 'title': 'Confidence', 'type': 'number'}}, 'required': ['text', 'label', 'confidence'], 'title': 'NEREntityFormat', 'type': 'object'}}, 'properties': {'entities': {'items': {'$ref': '#/$defs/NEREntityFormat'}, 'title': 'Entities', 'type': 'array'}}, 'required': ['entities'], 'title': 'NERResponse', 'type': 'object'}
Never output newlines, multiple spaces or explanatory text.
List of labels to use:
['BANK_ACCOUNT', 'DRIVERS_LICENSE', 'EMAIL_ADDRESS', 'IP_ADDRESS', 'PASSPORT', 'PCI', 'PERSON_NAME', 'PHONE_NUMBER', 'PRECISE_LOCATION', 'SALARY', 'SSN', 'USERNAME', 'VEHICLE_ID']
Pick exactly one label for each PII-sensitive substring/entity.
Below are descriptions of each label in the relevant language: <json string containing danish descriptions>.

user_prompt: Label all (possibly none) PII-entities in the following text: <input text>. In the order they appear.
```

### Appendix

#### OCR
| model            | dataset         | samples | mean_cer | mean_wer | ms/word | worst_cer | worst_cer_doc | worst_wer | worst_wer_doc |
|:-----------------|:----------------|--------:|---------:|---------:|-----------------:|:--------  | ------------- | :-------  | ---           |
| PaddleOCR-VL-1.5 | MassimoDS       |       5 | 0.118035 |   0.2873 |            82.92 |     0.278 | massimo_passport                | 0.75 | massimo_passport                  |
| PaddleOCR-VL-1.5 | dimmi-contracts |      53 | 0.212776 |   0.2562 |            34.63 |     0.655 | 42_jobpatruljen_1_bartender     | 0.57 | 24_kost_style_ernaeringsassistent |
| PaddleOCR-VL-1.5 | dimmi-cvs       |      60 | 0.251514 |   0.3085 |            95.54 |     1.049 | cv_44_Museumskurator            | 1.00 | cv_46_Lydteknikker                |
| PaddleOCR-VL-1.5 | dimmi-diplomas  |      20 | 0.435280 |   0.5141 |            36.56 |     2.259 | diploma_07_perspective_warp     | 2.85 | diploma_07_perspective_warp       |
| PaddleOCR-VL-1.5 | dimmi-passport  |      24 | 0.695867 |   0.7917 |           133.34 |     4.540 | 15126                           | 1.00 | 15126                             |
| Tesseract        | MassimoDS       |       5 | 0.154972 |   0.4180 |             7.15 |     0.436 | massimo_passport                | 1.03 | massimo_passport |
| Tesseract        | dimmi-contracts |      53 | 0.150779 |   0.5012 |             7.55 |     0.550 | 23_kost_style_kokkenleder       | 0.95 | 10_bygge_murer |
| Tesseract        | dimmi-cvs       |      60 | 0.209388 |   0.7549 |            11.58 |     0.934 | cv_33_UX_Designer               | 1.81 | cv_45_Sommelier |
| Tesseract        | dimmi-diplomas  |      20 | 0.446126 |   0.6644 |            52.29 |     1.300 | phd_diploma_07_perspective_warp | 1.71 | phd_diploma_07_perspective_warp |
| Tesseract        | dimmi-passport  |      24 | 0.719053 |   0.9620 |             4.67 |     1.000 | 385213                          | 1.19 | 383623 |
| qwen3.5-397b     | MassimoDS       |       5 | 0.142722 |   0.2871 |            71.24 |     0.375 | massimo_curriculum_vitae        | 0.61 | massimo_passport    |
| qwen3.5-397b     | dimmi-contracts |      53 | 0.079242 |   0.2661 |            52.67 |     0.419 | 10_bygge_murer                  | 0.52 | 11_bygge_elektriker |
| qwen3.5-397b     | dimmi-cvs       |      60 | 0.082453 |   0.3298 |           121.84 |     0.464 | cv_37_Pilot                     | 1.05 | cv_43_Marinbiolog   |
| qwen3.5-397b     | dimmi-passport  |      24 | 0.205987 |   0.5064 |           173.25 |     0.495 | 379593                          | 0.92 | 71640               |

#### Redaction

##### Performance per label

Note: Any labels which do not appear in the dataset has been left out.

| model                              | label            |   tp |    fp |   fn |    tn |   precision |   recall |
|:-----------------------------------|:-----------------|-----:|------:|-----:|------:|------------:|---------:|
| qwen3.5-397b                       | BANK_ACCOUNT     |    6 |    40 |    0 | 85113 |   0.130435  | 1        |
| qwen-235b                          | BANK_ACCOUNT     |    5 |     8 |    1 | 85145 |   0.384615  | 0.833333 |
| eu-pii-safeguard_biased_-11.5      | BANK_ACCOUNT     |    3 |   179 |    3 | 84974 |   0.0164835 | 0.5      |
| gliner-pii_th_0.0002               | BANK_ACCOUNT     |    3 |    33 |    3 | 85120 |   0.0833333 | 0.5      |
| qwen-235b                          | DRIVERS_LICENSE  |    2 |     4 |    1 | 85152 |   0.333333  | 0.666667 |
| qwen3.5-397b                       | DRIVERS_LICENSE  |    1 |     5 |    2 | 85151 |   0.166667  | 0.333333 |
| eu-pii-safeguard_biased_-11.5      | DRIVERS_LICENSE  |    0 |     8 |    3 | 85148 |   0         | 0        |
| gliner-pii_th_0.0002               | DRIVERS_LICENSE  |    0 |    81 |    3 | 85075 |   0         | 0        |
| eu-pii-safeguard_biased_-11.5      | EMAIL_ADDRESS    |  137 |   622 |   10 | 84390 |   0.180501  | 0.931973 |
| qwen3.5-397b                       | EMAIL_ADDRESS    |  135 |     4 |   12 | 85008 |   0.971223  | 0.918367 |
| qwen-235b                          | EMAIL_ADDRESS    |  129 |     4 |   18 | 85008 |   0.969925  | 0.877551 |
| gliner-pii_th_0.0002               | EMAIL_ADDRESS    |   27 |   199 |  120 | 84813 |   0.119469  | 0.183673 |
| qwen3.5-397b                       | PASSPORT         |   28 |    18 |    1 | 85112 |   0.608696  | 0.965517 |
| eu-pii-safeguard_biased_-11.5      | PASSPORT         |   18 |    54 |   11 | 85076 |   0.25      | 0.62069  |
| qwen-235b                          | PASSPORT         |   16 |    28 |   13 | 85102 |   0.363636  | 0.551724 |
| gliner-pii_th_0.0002               | PASSPORT         |   13 |   263 |   16 | 84867 |   0.0471014 | 0.448276 |
| qwen3.5-397b                       | PERSON_NAME      |  971 |   152 |   61 | 83975 |   0.864648  | 0.940891 |
| gliner-pii_th_0.0002               | PERSON_NAME      |  967 | 14409 |   65 | 69718 |   0.0628902 | 0.937016 |
| eu-pii-safeguard_biased_-11.5      | PERSON_NAME      |  957 |  3778 |   75 | 80349 |   0.202112  | 0.927326 |
| qwen-235b                          | PERSON_NAME      |  552 |    60 |  480 | 84067 |   0.901961  | 0.534884 |
| qwen3.5-397b                       | PHONE_NUMBER     |  785 |   116 |    2 | 84256 |   0.871254  | 0.997459 |
| eu-pii-safeguard_biased_-11.5      | PHONE_NUMBER     |  772 |   198 |   15 | 84174 |   0.795876  | 0.98094  |
| gliner-pii_th_0.0002               | PHONE_NUMBER     |  772 |   234 |   15 | 84138 |   0.767396  | 0.98094  |
| qwen-235b                          | PHONE_NUMBER     |  667 |     8 |  120 | 84364 |   0.988148  | 0.847522 |
| eu-pii-safeguard_biased_-11.5      | PRECISE_LOCATION | 1116 |  7743 |   40 | 76260 |   0.125974  | 0.965398 |
| qwen3.5-397b                       | PRECISE_LOCATION | 1060 |   660 |   96 | 83343 |   0.616279  | 0.916955 |
| gliner-pii_th_0.0002               | PRECISE_LOCATION | 1038 |  6887 |  118 | 77116 |   0.130978  | 0.897924 |
| qwen-235b                          | PRECISE_LOCATION |  666 |    87 |  490 | 83916 |   0.884462  | 0.576125 |
| qwen3.5-397b                       | SALARY           |  198 |   268 |   28 | 84665 |   0.424893  | 0.876106 |
| gliner-pii_th_0.0002               | SALARY           |  189 |  6115 |   37 | 78818 |   0.029981  | 0.836283 |
| qwen-235b                          | SALARY           |  150 |   105 |   76 | 84828 |   0.588235  | 0.663717 |
| eu-pii-safeguard_biased_-11.5      | SALARY           |  148 |   310 |   78 | 84623 |   0.323144  | 0.654867 |
| qwen3.5-397b                       | SSN              |  127 |    44 |    9 | 84979 |   0.74269   | 0.933824 |
| eu-pii-safeguard_biased_-11.5      | SSN              |  120 |    98 |   16 | 84925 |   0.550459  | 0.882353 |
| qwen-235b                          | SSN              |  116 |    71 |   20 | 84952 |   0.620321  | 0.852941 |
| gliner-pii_th_0.0002               | SSN              |  106 |   158 |   30 | 84865 |   0.401515  | 0.779412 |
| qwen3.5-397b                       | USERNAME         |   21 |    15 |    9 | 85114 |   0.583333  | 0.7      |
| qwen-235b                          | USERNAME         |   17 |     0 |   13 | 85129 |   1         | 0.566667 |
| eu-pii-safeguard_biased_-11.5      | USERNAME         |    0 |     5 |   30 | 85124 |   0         | 0        |
| gliner-pii_th_0.0002               | USERNAME         |    0 |    32 |   30 | 85097 |   0         | 0        |

##### Worst documents

| model                         | dataset   | document                      |   precision |   recall |
|:------------------------------|:----------|:----------------------------  |------------:|---------:|
| eu-pii-safeguard_biased_-11.5 | contracts | 11_bygge_elektriker           |  0.509      | 0.848    |
| eu-pii-safeguard_biased_-11.5 | cvs       | cv_47_Byplanlægger            |  0.360      | 0.818    |
| eu-pii-safeguard_biased_-11.5 | diplomas  | diploma_05_low_contrast_faded |  0.032      | 0.800    |
| eu-pii-safeguard_biased_-11.5 | passports | 384732                        |  0.133      | 0.666    |
| gliner-pii_th_0.0002          | contracts | 11_bygge_elektriker           |  0.178      | 0.848    |
| gliner-pii_th_0.0002          | cvs       | cv_47_Byplanlægger            |  0.409      | 0.818    |
| gliner-pii_th_0.0002          | diplomas  | diploma_05_low_contrast_faded |  0.015      | 0.800    |
| gliner-pii_th_0.0002          | passports | 384732                        |  0.093      | 0.666    |
| qwen3.5-397b                  | contracts | 37_ku_1_postdoc               |  0.897      | 0.578    |
| qwen3.5-397b                  | cvs       | cv_36_Dyrlæge                 |  0.900      | 0.900    |
| qwen3.5-397b                  | diplomas  | diploma_07_perspective_warp   |  0.684      | 0.722    |
| qwen3.5-397b                  | passports | 384732                        |  0.375      | 0.500    |
