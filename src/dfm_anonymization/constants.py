from bidict import bidict

ISO_LABEL_BANK_ACCOUNT = "BANK_ACCOUNT"  # !
ISO_LABEL_DRIVERS_LICENSE = "DRIVERS_LICENSE"  #
ISO_LABEL_EMAIL_ADDRESS = "EMAIL_ADDRESS"  #
ISO_LABEL_IP_ADDRESS = "IP_ADDRESS"  # !
ISO_LABEL_OTHER_PII = "OTHER_PII"  # !
ISO_LABEL_PASSPORT = "PASSPORT"  # !
ISO_LABEL_PCI = "PCI"  #
ISO_LABEL_PERSON_NAME = "PERSON_NAME"  #
ISO_LABEL_PHONE_NUMBER = "PHONE_NUMBER"  #
ISO_LABEL_PRECISE_LOCATION = "PRECISE_LOCATION"  #
ISO_LABEL_SSN = "SSN"  #
ISO_LABEL_USERNAME = "USERNAME"  # !
ISO_LABEL_VEHICLE_ID = "VEHICLE_ID"  # !
# PRIVATE_URL

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
    ISO_LABEL_SSN,
    ISO_LABEL_USERNAME,
    ISO_LABEL_VEHICLE_ID,
}

LABEL_DESCRIPTIONS_EN = bidict(
    {
        ISO_LABEL_BANK_ACCOUNT: "bank account number",
        ISO_LABEL_DRIVERS_LICENSE: "drivers license number",
        ISO_LABEL_EMAIL_ADDRESS: (
            "private email address (public/official emails not included) "
            "- e.g. hjohnson@hotmail.com, but not contact@cbs.com"
        ),
        ISO_LABEL_IP_ADDRESS: "IP address",
        ISO_LABEL_OTHER_PII: "other personal identifiable information",
        ISO_LABEL_PASSPORT: "passport number",
        ISO_LABEL_PCI: "payment card information",
        ISO_LABEL_PERSON_NAME: (
            "name of a person (whenever it's possible to infer "
            "this persons identity from this name) - e.g. full names - "
            "with the exception of public personas which will not be considered "
            "in this category (e.g. people with a wikipedia page)"
        ),
        ISO_LABEL_PHONE_NUMBER: "telephone number - any country",
        ISO_LABEL_PRECISE_LOCATION: (
            "precise location information - e.g. streetname with house number or "
            "coordinates"
        ),
        ISO_LABEL_SSN: "social security number",
        ISO_LABEL_USERNAME: "username",
        ISO_LABEL_VEHICLE_ID: "vehicle identification number",
    }
)

LABEL_DESCRIPTIONS_DA = bidict(
    {
        ISO_LABEL_BANK_ACCOUNT: "bankkontonummer",
        ISO_LABEL_DRIVERS_LICENSE: "kørekortnummer",
        ISO_LABEL_EMAIL_ADDRESS: (
            "private email-adresser (men ikke offentlige/officielle email-adresser) "
            "- f.eks hbergman@gmail.com, men ikke kontakt@dr.dk"
        ),
        ISO_LABEL_IP_ADDRESS: "IP-adresse",
        ISO_LABEL_OTHER_PII: "andet person-identificerende information",
        ISO_LABEL_PASSPORT: "pasnummer",
        ISO_LABEL_PCI: "betalingskortoplysninger",
        ISO_LABEL_PERSON_NAME: (
            "navn på en person (når dette navn gør det muligt at identificere "
            "personen) - f.eks. navn + efternavn - med den undtagelse at offentlige "
            "personer ikke betragtes under denne kategori "
            "(f.eks. folk som har en side på wikipedia)"
        ),
        ISO_LABEL_PHONE_NUMBER: "telefonnummer - alle landekoder",
        ISO_LABEL_PRECISE_LOCATION: "præcis lokation - gadenavn med husnummer eller koordinater",
        ISO_LABEL_SSN: "CPR-nummer",
        ISO_LABEL_USERNAME: "brugernavn",
        ISO_LABEL_VEHICLE_ID: "registreringsnummer / nummerplade",
    }
)

GLINER_MODELS = ["nvidia/gliner-pii", "urchade/gliner_multi-v2.1"]

NER_MODELS = ["nbailab-base-ner-scandi"]

REGEX_COMPATIBLE_LABELS = {
    ISO_LABEL_BANK_ACCOUNT,
    ISO_LABEL_DRIVERS_LICENSE,
    ISO_LABEL_EMAIL_ADDRESS,
    ISO_LABEL_PASSPORT,
    ISO_LABEL_PHONE_NUMBER,
    ISO_LABEL_PCI,
    ISO_LABEL_IP_ADDRESS,
    ISO_LABEL_VEHICLE_ID,
    ISO_LABEL_SSN,
    ISO_LABEL_PRECISE_LOCATION,
}

MODEL_LABEL_MAP: dict[str, dict[str, str]] = {
    "nbailab-base-ner-scandi": {
        "LOC": ISO_LABEL_PRECISE_LOCATION,
        "PER": ISO_LABEL_PERSON_NAME,
    },
    "presidio": {
        "PERSON": ISO_LABEL_PERSON_NAME,
        "LOCATION": ISO_LABEL_PRECISE_LOCATION,
        "PHONE_NUMBER": ISO_LABEL_PHONE_NUMBER,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL_ADDRESS,
    },
    "scrambletron": {
        "PERSON": ISO_LABEL_PERSON_NAME,
        "LOCATION": ISO_LABEL_PRECISE_LOCATION,
        "PHONE_NUMBER": ISO_LABEL_PHONE_NUMBER,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL_ADDRESS,
        "DK_DRIVERS_LICENSE": ISO_LABEL_DRIVERS_LICENSE,
        "IBAN_CODE": ISO_LABEL_BANK_ACCOUNT,
        "DK_SSN": ISO_LABEL_SSN,
        "CREDIT_CARD": ISO_LABEL_PCI,
    },
    "regex": {k: k for k in REGEX_COMPATIBLE_LABELS},
}

DATASET_LABEL_MAP: dict[str, dict[str, str]] = {
    "dane": {
        # "LOC": ISO_LABEL_PRECISE_LOCATION,
        "PER": ISO_LABEL_PERSON_NAME,
    },
    "dansk-ner": {
        "PERSON": ISO_LABEL_PERSON_NAME,
    },
    "gen_presidio": {
        "DRIVERS_LICENSE": ISO_LABEL_DRIVERS_LICENSE,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL_ADDRESS,
        "PCI": ISO_LABEL_PCI,
        "PERSON_NAME": ISO_LABEL_PERSON_NAME,
        "PHONE_NUMBER": ISO_LABEL_PHONE_NUMBER,
        "PRECISE_LOCATION": ISO_LABEL_PRECISE_LOCATION,
        "SSN": ISO_LABEL_SSN,
    },
    "gen_prompt": {
        "BANK_ACCOUNT": ISO_LABEL_BANK_ACCOUNT,
        "DRIVERS_LICENSE": ISO_LABEL_DRIVERS_LICENSE,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL_ADDRESS,
        "IP_ADDRESS": ISO_LABEL_IP_ADDRESS,
        "OTHER_PII": ISO_LABEL_OTHER_PII,
        "PASSPORT": ISO_LABEL_PASSPORT,
        "PCI": ISO_LABEL_PCI,
        "PERSON_NAME": ISO_LABEL_PERSON_NAME,
        "PHONE_NUMBER": ISO_LABEL_PHONE_NUMBER,
        "PRECISE_LOCATION": ISO_LABEL_PRECISE_LOCATION,
        "SSN": ISO_LABEL_SSN,
        "USERNAME": ISO_LABEL_USERNAME,
        "VEHICLE_ID": ISO_LABEL_VEHICLE_ID,
    },
    "openpii-1m-da": {
        # "DATE",
        "GIVENNAME": ISO_LABEL_PERSON_NAME,
        "SURNAME": ISO_LABEL_PERSON_NAME,
        "EMAIL": ISO_LABEL_EMAIL_ADDRESS,
        "CITY": ISO_LABEL_PRECISE_LOCATION,
        # "TITLE",
        "TELEPHONENUM": ISO_LABEL_PHONE_NUMBER,
        # "AGE",
        "STREET": ISO_LABEL_PRECISE_LOCATION,
        "BUILDINGNUM": ISO_LABEL_PRECISE_LOCATION,
        "ZIPCODE": ISO_LABEL_PRECISE_LOCATION,
        "IDCARDNUM": ISO_LABEL_OTHER_PII,
        "CREDITCARDNUMBER": ISO_LABEL_PCI,
        "DRIVERLICENSENUM": ISO_LABEL_DRIVERS_LICENSE,
        # "GENDER",
        # "TAXNUM",
        # "SEX",
        "SOCIALNUM": ISO_LABEL_SSN,
        "PASSPORTNUM": ISO_LABEL_PASSPORT,
    },
}

DATASET_LABEL_MAP = DATASET_LABEL_MAP | {
    "openpii-1m-da-s": DATASET_LABEL_MAP["openpii-1m-da"]
}
