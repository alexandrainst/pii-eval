from typing import Dict, Optional

LABEL_ORG_DA = "organisation"
LABEL_LOC_DA = "lokation"
LABEL_PER_DA = "personnavn"
LABEL_URL_DA = "webadresse(url)"
LABEL_PHONE_DA = "telefonnummer"
LABEL_EMAIL_DA = "email"
LABEL_DRIVERS_LICENSE_DA = "kørekortnummer"
LABEL_DATE_TIME_DA = "dato/tid"
LABEL_IBAN_DA = "cvv-nummer"
LABEL_SSN_DA = "cpr-nummer"
LABEL_CARD_NUMBER_DA = "betalingskortnummer"

GLINER_MODELS = [
    "nvidia/gliner-pii",
    "urchade/gliner_multi-v2.1"
]

NER_MODELS = [
    "scandiner"
]

MODEL_LABEL_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "scandiner": {
        "ORG": LABEL_ORG_DA,
        "LOC": LABEL_LOC_DA,
        "PER": LABEL_PER_DA,
        "MISC": None
    },
    "custom_presidio": {
        "ORGANIZATION": LABEL_ORG_DA,
        "PERSON": LABEL_PER_DA,
        "LOCATION": LABEL_LOC_DA,
        "URL": LABEL_URL_DA,
        "PHONE_NUMBER": LABEL_PHONE_DA,
        "DATE_TIME": LABEL_DATE_TIME_DA,
        "EMAIL_ADDRESS": LABEL_EMAIL_DA
    },
    "scrambletron": {
        "ORGANIZATION": LABEL_ORG_DA,
        "PERSON": LABEL_PER_DA,
        "LOCATION": LABEL_LOC_DA,
        "URL": LABEL_URL_DA,
        "PHONE_NUMBER": LABEL_PHONE_DA,
        "DATE_TIME": LABEL_DATE_TIME_DA,
        "EMAIL_ADDRESS": LABEL_EMAIL_DA,
        "DK_DRIVERS_LICENSE": LABEL_DRIVERS_LICENSE_DA,
        "IBAN_CODE": LABEL_IBAN_DA,
        "DK_SSN": LABEL_SSN_DA,
        "CREDIT_CARD": LABEL_CARD_NUMBER_DA
    }
}

DATASET_LABEL_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "dane": {
        "ORG": LABEL_ORG_DA,
        "MISC": None,
        "LOC": LABEL_LOC_DA,
        "PER": LABEL_PER_DA
    },
    "dansk_ner": {
        "WORK OF ART": None,
        "MONEY": None,
        "QUANTITY": None,
        "DATE": None, # too low quality
        "GPE": None,
        "TIME": None, # too low quality
        "PERSON": LABEL_PER_DA,
        "CARDINAL": None,
        "PERCENT": None,
        "FACILITY": None,
        "LANGUAGE": None,
        "ORDINAL": None,
        "NORP": None,
        "EVENT": None,
    },
    "gen_address": {
        "address": LABEL_LOC_DA
    },
    "gen_card-expiry": {
        "card-expiry": LABEL_DATE_TIME_DA
    },
    "gen_card-number": {
        "card-number": LABEL_CARD_NUMBER_DA
    },
    "gen_cpr": {
        "cpr": LABEL_SSN_DA
    },
    "gen_cvv": {
        "cvv": LABEL_IBAN_DA
    },
    "gen_dob": {
        "dob": LABEL_DATE_TIME_DA
    },
    "gen_driver-license": {
        "driver-license": LABEL_DRIVERS_LICENSE_DA
    },
    "gen_email": {
        "email": LABEL_EMAIL_DA
    },
    "gen_organization": {
        "organization": LABEL_ORG_DA
    },
    "gen_person": {
        "person": LABEL_PER_DA
    },
    "gen_phone-number": {
        "phone-number": LABEL_PHONE_DA
    },
    "gen_website": {
        "website": LABEL_URL_DA
    }
}
