from typing import Dict, Optional


LABEL_TRANSLATIONS: Dict[str, Dict[str, Optional[str]]] = {
    "dane": {
        "ORG": "organisation",
        "MISC": None,
        "LOC": "lokation",
        "PER": "personnavn"
    },
    "dansk_ner": {
        "WORK OF ART": None,
        "MONEY": None,
        "QUANTITY": None,
        "DATE": None,
        "GPE": None,
        "TIME": None,
        "PERSON": "personnavn",
        "CARDINAL": None,
        "PERCENT": None,
        "FACILITY": None,
        "LANGUAGE": None,
        "ORDINAL": None,
        "NORP": None,
        "EVENT": None,
    },
    "scrambletron_gen/da_address": {
        "address": "adresse"
    },
    "scrambletron_gen/da_cpr": {
        "cpr": "cpr-nummer"
    }
}
