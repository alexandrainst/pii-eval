from typing import Dict, Optional
from bidict import bidict

LABEL_DESCRIPTIONS_DA = bidict({
    "ORG": "organisation",
    "LOC": "lokation",
    "PER": "personnavn",
    "URL": "webadresse(url)",
    "PHONE": "telefonnummer",
    "EMAIL": "email",
    "DRIVERS_LICENSE": "kørekortnummer",
    "DATE_TIME": "dato/tid",
    "IBAN": "cvv-nummer",
    "SSN": "cpr-nummer",
    "CARD_NUMBER": "betalingskortnummer",
})

GLINER_MODELS = [
    "nvidia/gliner-pii",
    "urchade/gliner_multi-v2.1"
]

NER_MODELS = [
    "nbailab-base-ner-scandi"
]

MODEL_LABEL_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "nbailab-base-ner-scandi": {
        "ORG": LABEL_DESCRIPTIONS_DA["ORG"],
        "LOC": LABEL_DESCRIPTIONS_DA["LOC"],
        "PER": LABEL_DESCRIPTIONS_DA["PER"],
        "MISC": None # mostly not relevant in pii context
    },
    "presidio": {
        "ORGANIZATION": LABEL_DESCRIPTIONS_DA["ORG"],
        "PERSON": LABEL_DESCRIPTIONS_DA["PER"],
        "LOCATION": LABEL_DESCRIPTIONS_DA["LOC"],
        "URL": LABEL_DESCRIPTIONS_DA["URL"],
        "PHONE_NUMBER": LABEL_DESCRIPTIONS_DA["PHONE"],
        "DATE_TIME": LABEL_DESCRIPTIONS_DA["DATE_TIME"],
        "EMAIL_ADDRESS": LABEL_DESCRIPTIONS_DA["EMAIL"]
    },
    "scrambletron": {
        "ORGANIZATION": LABEL_DESCRIPTIONS_DA["ORG"],
        "PERSON": LABEL_DESCRIPTIONS_DA["PER"],
        "LOCATION": LABEL_DESCRIPTIONS_DA["LOC"],
        "URL": LABEL_DESCRIPTIONS_DA["URL"],
        "PHONE_NUMBER": LABEL_DESCRIPTIONS_DA["PHONE"],
        "DATE_TIME": LABEL_DESCRIPTIONS_DA["DATE_TIME"],
        "EMAIL_ADDRESS": LABEL_DESCRIPTIONS_DA["EMAIL"],
        "DK_DRIVERS_LICENSE": LABEL_DESCRIPTIONS_DA["DRIVERS_LICENSE"],
        "IBAN_CODE": LABEL_DESCRIPTIONS_DA["IBAN"],
        "DK_SSN": LABEL_DESCRIPTIONS_DA["SSN"],
        "CREDIT_CARD": LABEL_DESCRIPTIONS_DA["CARD_NUMBER"]
    }
}

DATASET_LABEL_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "dane": {
        "ORG": LABEL_DESCRIPTIONS_DA["ORG"],
        "MISC": None, # mostly not relevant in pii context
        "LOC": LABEL_DESCRIPTIONS_DA["LOC"],
        "PER": LABEL_DESCRIPTIONS_DA["PER"]
    },
    "dansk_ner": {
        "WORK OF ART": None,
        "MONEY": None,
        "QUANTITY": None,
        "DATE": None, # mostly not precise enough for pii in this dataset
        "GPE": None,
        "TIME": None, # mostly not precise enough for pii in this dataset
        "PERSON": LABEL_DESCRIPTIONS_DA["PER"],
        "CARDINAL": None,
        "PERCENT": None,
        "FACILITY": None,
        "LANGUAGE": None,
        "ORDINAL": None,
        "NORP": None,
        "EVENT": None,
    },
    "gen_address": {
        "address": LABEL_DESCRIPTIONS_DA["LOC"]
    },
    "gen_card-expiry": {
        "card-expiry": LABEL_DESCRIPTIONS_DA["DATE_TIME"]
    },
    "gen_card-number": {
        "card-number": LABEL_DESCRIPTIONS_DA["CARD_NUMBER"]
    },
    "gen_cpr": {
        "cpr": LABEL_DESCRIPTIONS_DA["SSN"]
    },
    "gen_cvv": {
        "cvv": LABEL_DESCRIPTIONS_DA["IBAN"]
    },
    "gen_dob": {
        "dob": LABEL_DESCRIPTIONS_DA["DATE_TIME"]
    },
    "gen_driver-license": {
        "driver-license": LABEL_DESCRIPTIONS_DA["DRIVERS_LICENSE"]
    },
    "gen_email": {
        "email": LABEL_DESCRIPTIONS_DA["EMAIL"]
    },
    "gen_organization": {
        "organization": LABEL_DESCRIPTIONS_DA["ORG"]
    },
    "gen_person": {
        "person": LABEL_DESCRIPTIONS_DA["PER"]
    },
    "gen_phone-number": {
        "phone-number": LABEL_DESCRIPTIONS_DA["PHONE"]
    },
    "gen_website": {
        "website": LABEL_DESCRIPTIONS_DA["URL"]
    }
}
