from typing import Dict, Optional

LABEL_ORG_DA = "organisation"
LABEL_LOC_DA = "lokation"
LABEL_PER_DA = "personnavn"

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
        "DATE": None,
        "GPE": None,
        "TIME": None,
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
        "card-expiry": "betalingskortudløbsdato"
    },
    "gen_card-number": {
        "card-number": "betalingskortnummer"
    },
    "gen_cpr": {
        "cpr": "cpr-nummer"
    },
    "gen_cvv": {
        "cvv": "cvv-nummer"
    },
    "gen_dob": {
        "dob": "fødselsdato"
    },
    "gen_driver-license": {
        "driver-license": "kørekortnummer"
    },
    "gen_email": {
        "email": "email"
    },
    "gen_organization": {
        "organization": LABEL_ORG_DA
    },
    "gen_person": {
        "person": LABEL_PER_DA
    },
    "gen_phone-number": {
        "phone-number": "telefonnummer"
    },
    "gen_website": {
        "website": "webadresse(url)"
    }
}
