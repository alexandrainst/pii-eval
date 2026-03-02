from bidict import bidict

ISO_LABEL_BANK_ACCOUNT = "BANK_ACCOUNT"
ISO_LABEL_DRIVERS_LICENSE = "DRIVERS_LICENSE"
ISO_LABEL_EMAIL_ADDRESS = "EMAIL_ADDRESS"
ISO_LABEL_IP_ADDRESS = "IP_ADDRESS"
ISO_LABEL_OTHER_PII = "OTHER_PII"
ISO_LABEL_PASSPORT = "PASSPORT"
ISO_LABEL_PCI = "PCI"
ISO_LABEL_PERSON_NAME = "PERSON_NAME"
ISO_LABEL_PHONE_NUMBER = "PHONE_NUMBER"
ISO_LABEL_PRECISE_LOCATION = "PRECISE_LOCATION"
ISO_LABEL_PRIVATE_URL = "PRIVATE_URL"
ISO_LABEL_SSN = "SSN"
ISO_LABEL_USERNAME = "USERNAME"
ISO_LABEL_VEHICLE_ID = "VEHICLE_ID"

ALL_ISO_LABELS = {
    ISO_LABEL_BANK_ACCOUNT,
    ISO_LABEL_DRIVERS_LICENSE,
    ISO_LABEL_EMAIL_ADDRESS,
    ISO_LABEL_IP_ADDRESS,
    ISO_LABEL_OTHER_PII,
    ISO_LABEL_PASSPORT,
    ISO_LABEL_PCI,
    ISO_LABEL_PERSON_NAME,
    ISO_LABEL_PHONE_NUMBER,
    ISO_LABEL_PRECISE_LOCATION,
    ISO_LABEL_PRIVATE_URL,
    ISO_LABEL_SSN,
    ISO_LABEL_USERNAME,
    ISO_LABEL_VEHICLE_ID,
}

LABEL_DESCRIPTIONS_EN = bidict(
    {
        ISO_LABEL_BANK_ACCOUNT: "bank account number",
        ISO_LABEL_DRIVERS_LICENSE: "drivers license number",
        ISO_LABEL_EMAIL_ADDRESS: "email address",
        ISO_LABEL_IP_ADDRESS: "IP address",
        ISO_LABEL_OTHER_PII: "other personal identifiable information",
        ISO_LABEL_PASSPORT: "passport number",
        ISO_LABEL_PCI: "payment card information",
        ISO_LABEL_PERSON_NAME: "name of a person",
        ISO_LABEL_PHONE_NUMBER: "phone number",
        ISO_LABEL_PRECISE_LOCATION: "precise location information",
        ISO_LABEL_PRIVATE_URL: "private URL",
        ISO_LABEL_SSN: "social security number",
        ISO_LABEL_USERNAME: "username",
        ISO_LABEL_VEHICLE_ID: "vehicle identification number",
    }
)

LABEL_DESCRIPTIONS_DA = bidict(
    {
        ISO_LABEL_BANK_ACCOUNT: "bankkontonummer",
        ISO_LABEL_DRIVERS_LICENSE: "kørekortnummer",
        ISO_LABEL_EMAIL_ADDRESS: "email-adresse",
        ISO_LABEL_IP_ADDRESS: "IP-adresse",
        ISO_LABEL_OTHER_PII: "andet person identificerende",
        ISO_LABEL_PASSPORT: "pasnummer",
        ISO_LABEL_PCI: "betalingskortoplysninger",
        ISO_LABEL_PERSON_NAME: "personnavn",
        ISO_LABEL_PHONE_NUMBER: "telefonnummer",
        ISO_LABEL_PRECISE_LOCATION: "præcis lokation",
        ISO_LABEL_PRIVATE_URL: "privat url",
        ISO_LABEL_SSN: "CPR-nummer",
        ISO_LABEL_USERNAME: "brugernavn",
        ISO_LABEL_VEHICLE_ID: "registreringsnummer / nummerplade",
    }
)

GLINER_MODELS = ["nvidia/gliner-pii", "urchade/gliner_multi-v2.1"]

NER_MODELS = ["nbailab-base-ner-scandi"]

MODEL_LABEL_MAP: dict[str, dict[str, str]] = {
    "nbailab-base-ner-scandi": {
        "LOC": ISO_LABEL_PRECISE_LOCATION,
        "PER": ISO_LABEL_PERSON_NAME,
    },
    "presidio": {
        "PERSON": ISO_LABEL_PERSON_NAME,
        "LOCATION": ISO_LABEL_PRECISE_LOCATION,
        "URL": ISO_LABEL_PRIVATE_URL,
        "PHONE_NUMBER": ISO_LABEL_PHONE_NUMBER,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL_ADDRESS,
    },
    "scrambletron": {
        "PERSON": ISO_LABEL_PERSON_NAME,
        "LOCATION": ISO_LABEL_PRECISE_LOCATION,
        "URL": ISO_LABEL_PRIVATE_URL,
        "PHONE_NUMBER": ISO_LABEL_PHONE_NUMBER,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL_ADDRESS,
        "DK_DRIVERS_LICENSE": ISO_LABEL_DRIVERS_LICENSE,
        "IBAN_CODE": ISO_LABEL_BANK_ACCOUNT,
        "DK_SSN": ISO_LABEL_SSN,
        "CREDIT_CARD": ISO_LABEL_PCI,
    },
}

DATASET_LABEL_MAP: dict[str, dict[str, str]] = {
    "dane": {
        "LOC": ISO_LABEL_PRECISE_LOCATION,
        "PER": ISO_LABEL_PERSON_NAME,
    },
    "dansk-ner": {
        "PERSON": ISO_LABEL_PERSON_NAME,
    },
}
