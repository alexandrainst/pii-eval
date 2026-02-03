from typing import Dict, Optional
from bidict import bidict

ISO_LABEL_ORGANIZATION = "ORGANIZATION"
ISO_LABEL_LOCATION = "LOCATION"
ISO_LABEL_PERSON = "PERSON"
ISO_LABEL_URL = "URL"
ISO_LABEL_PHONE = "PHONE"
ISO_LABEL_EMAIL = "EMAIL"
ISO_LABEL_DRIVERS_LICENSE = "DRIVERS_LICENSE"
ISO_LABEL_DATE_TIME = "DATE_TIME"
ISO_LABEL_IBAN = "IBAN"
ISO_LABEL_SSN = "SSN"
ISO_LABEL_CARD_NUMBER = "CARD_NUMBER"

LABEL_DESCRIPTIONS_DA = bidict(
    {
        ISO_LABEL_ORGANIZATION: "organisation",
        ISO_LABEL_LOCATION: "lokation",
        ISO_LABEL_PERSON: "personnavn",
        ISO_LABEL_URL: "webadresse(url)",
        ISO_LABEL_PHONE: "telefonnummer",
        ISO_LABEL_EMAIL: "email",
        ISO_LABEL_DRIVERS_LICENSE: "kørekortnummer",
        ISO_LABEL_DATE_TIME: "dato/tid",
        ISO_LABEL_IBAN: "cvv-nummer",
        ISO_LABEL_SSN: "cpr-nummer",
        ISO_LABEL_CARD_NUMBER: "betalingskortnummer",
    }
)

GLINER_MODELS = ["nvidia/gliner-pii", "urchade/gliner_multi-v2.1"]

NER_MODELS = ["nbailab-base-ner-scandi"]

MODEL_LABEL_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "nbailab-base-ner-scandi": {
        "ORG": ISO_LABEL_ORGANIZATION,
        "LOC": ISO_LABEL_LOCATION,
        "PER": ISO_LABEL_PERSON,
        "MISC": None,  # mostly not relevant in pii context
    },
    "presidio": {
        "ORGANIZATION": ISO_LABEL_ORGANIZATION,
        "PERSON": ISO_LABEL_PERSON,
        "LOCATION": ISO_LABEL_LOCATION,
        "URL": ISO_LABEL_URL,
        "PHONE_NUMBER": ISO_LABEL_PHONE,
        "DATE_TIME": ISO_LABEL_DATE_TIME,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL,
    },
    "scrambletron": {
        "ORGANIZATION": ISO_LABEL_ORGANIZATION,
        "PERSON": ISO_LABEL_PERSON,
        "LOCATION": ISO_LABEL_LOCATION,
        "URL": ISO_LABEL_URL,
        "PHONE_NUMBER": ISO_LABEL_PHONE,
        "DATE_TIME": ISO_LABEL_DATE_TIME,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL,
        "DK_DRIVERS_LICENSE": ISO_LABEL_DRIVERS_LICENSE,
        "IBAN_CODE": ISO_LABEL_IBAN,
        "DK_SSN": ISO_LABEL_SSN,
        "CREDIT_CARD": ISO_LABEL_CARD_NUMBER,
    },
}

DATASET_LABEL_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "dane": {
        "ORG": ISO_LABEL_ORGANIZATION,
        "MISC": None,  # mostly not relevant in pii context
        "LOC": ISO_LABEL_LOCATION,
        "PER": ISO_LABEL_PERSON,
    },
    "dansk-ner": {
        "WORK OF ART": None,
        "MONEY": None,
        "QUANTITY": None,
        "DATE": None,  # mostly not precise enough for pii in this dataset
        "GPE": None,
        "TIME": None,  # mostly not precise enough for pii in this dataset
        "PERSON": ISO_LABEL_PERSON,
        "CARDINAL": None,
        "PERCENT": None,
        "FACILITY": None,
        "LANGUAGE": None,
        "ORDINAL": None,
        "NORP": None,
        "EVENT": None,
    },
    "gen_address": {"address": ISO_LABEL_LOCATION},
    "gen_card-expiry": {"card-expiry": ISO_LABEL_DATE_TIME},
    "gen_card-number": {"card-number": ISO_LABEL_CARD_NUMBER},
    "gen_cpr": {"cpr": ISO_LABEL_SSN},
    "gen_cvv": {"cvv": ISO_LABEL_IBAN},
    "gen_dob": {"dob": ISO_LABEL_DATE_TIME},
    "gen_driver-license": {"driver-license": ISO_LABEL_DRIVERS_LICENSE},
    "gen_email": {"email": ISO_LABEL_EMAIL},
    "gen_organization": {"organization": ISO_LABEL_ORGANIZATION},
    "gen_person": {"person": ISO_LABEL_PERSON},
    "gen_phone-number": {"phone-number": ISO_LABEL_PHONE},
    "gen_website": {"website": ISO_LABEL_URL},
}
