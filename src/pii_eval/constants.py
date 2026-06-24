import re

from bidict import bidict

# Labels to consider including:
#
# DEVICE_ID (eu-pii-safeguard) (*)
# DOB (multiple) *
# MAC_ADDRESS (eu-pii-safeguard) *
# PASSWORD (eu-pii-safeguard) (*)
# PRIVATE_URL (*)
# TAX_NUMBER (openpii-1m) *
#
# Labels tentatively excluded:
#
# AGE (eu-pii-safeguard) (*)
# AMOUNT (eu-pii-safeguard) *
# COMPANY_NAME (eu-pii-safeguard) (*)
# CURRENCY (eu-pii-safeguard) *
# ETHNICITY (eu-pii-safeguard) (*)
# GENDER (eu-pii-safeguard) *
# HEALTH_CONDITION (eu-pii-safeguard) (*)
# JOB_TITLE (eu-pii-safeguard) (*)
# OTHER_PII "other personal identifiable information"
# POLITICAL_OPINION (eu-pii-safeguard) (*)
# PREFIX (eu-pii-safeguard) (*)
# RELIGION (eu-pii-safeguard) (*)
# SEXUAL_ORIENTATION (eu-pii-safeguard) (*)
# URL (eu-pii-safeguard) (*)
#
# * = detectable by regex (*) = potentially

ISO_LABEL_BANK_ACCOUNT = "BANK_ACCOUNT"
ISO_LABEL_DRIVERS_LICENSE = "DRIVERS_LICENSE"
ISO_LABEL_EMAIL_ADDRESS = "EMAIL_ADDRESS"
ISO_LABEL_IP_ADDRESS = "IP_ADDRESS"
ISO_LABEL_PASSPORT = "PASSPORT"
ISO_LABEL_PCI = "PCI"
ISO_LABEL_PERSON_NAME = "PERSON_NAME"
ISO_LABEL_PHONE_NUMBER = "PHONE_NUMBER"
ISO_LABEL_PRECISE_LOCATION = "PRECISE_LOCATION"
ISO_LABEL_SSN = "SSN"
ISO_LABEL_USERNAME = "USERNAME"
ISO_LABEL_VEHICLE_ID = "VEHICLE_ID"
ISO_LABEL_SALARY = "SALARY"


ALL_ISO_LABELS = {
    ISO_LABEL_BANK_ACCOUNT,
    ISO_LABEL_DRIVERS_LICENSE,
    ISO_LABEL_EMAIL_ADDRESS,
    ISO_LABEL_IP_ADDRESS,
    ISO_LABEL_PASSPORT,
    ISO_LABEL_PCI,
    ISO_LABEL_PERSON_NAME,
    ISO_LABEL_PHONE_NUMBER,
    ISO_LABEL_PRECISE_LOCATION,
    ISO_LABEL_SALARY,
    ISO_LABEL_SSN,
    ISO_LABEL_USERNAME,
    ISO_LABEL_VEHICLE_ID,
}

REGEX_COMPATIBLE_LABELS = {
    ISO_LABEL_BANK_ACCOUNT,
    ISO_LABEL_DRIVERS_LICENSE,
    ISO_LABEL_EMAIL_ADDRESS,
    ISO_LABEL_IP_ADDRESS,
    ISO_LABEL_PASSPORT,
    ISO_LABEL_PCI,
    ISO_LABEL_PHONE_NUMBER,
    ISO_LABEL_SSN,
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
        ISO_LABEL_SALARY: "salary - precise amount in some currency earned (monthly, yearly, etc.) by an employee",
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
        ISO_LABEL_IP_ADDRESS: "IP-adresser",
        ISO_LABEL_PASSPORT: "pasnummer",
        ISO_LABEL_PCI: "betalingskortoplysninger, CVV",
        ISO_LABEL_PERSON_NAME: (
            "navn på en person (når dette navn gør det muligt at identificere "
            "personen) - f.eks. navn + efternavn - med den undtagelse at offentlige "
            "personer ikke betragtes under denne kategori "
            "(f.eks. folk som har en side på wikipedia)"
        ),
        ISO_LABEL_PHONE_NUMBER: "telefonnummer (alle landekoder)",
        ISO_LABEL_PRECISE_LOCATION: "præcis lokation - gadenavn med husnummer eller koordinater",
        ISO_LABEL_SALARY: "løn - præcist beløb tjent (månedligt / årligt / etc.) af en medarbejder",
        ISO_LABEL_SSN: "CPR-nummer",
        ISO_LABEL_USERNAME: "brugernavne (fx. på online platforme)",
        ISO_LABEL_VEHICLE_ID: "registreringsnummer / nummerplade",
    }
)

GLINER_MODELS = ["nvidia/gliner-pii", "urchade/gliner_multi-v2.1"]

NER_MODELS = ["nbailab-base-ner-scandi"]

DATASET_LABEL_MAP: dict[str, dict[str, str]] = {
    "dane": {
        "LOC": ISO_LABEL_PRECISE_LOCATION,
        # "MISC",
        # "ORG",
        "PER": ISO_LABEL_PERSON_NAME,
    },
    "dansk-ner": {
        # "CARDINAL",
        # "DATE",
        # "EVENT",
        # "FACILITY",
        # "GPE",
        # "LANGUAGE",
        # "LAW",
        # "LOCATION": ISO_LABEL_PRECISE_LOCATION,
        # "MONEY",
        # "NORP",
        # "ORDINAL",
        # "ORGANIZATION",
        # "PERCENT",
        "PERSON": ISO_LABEL_PERSON_NAME,
        # "PRODUCT",
        # "QUANTITY",
        # "TIME",
        # "WORK OF ART"
    },
    "dimmi": {
        "BANK_ACCOUNT": ISO_LABEL_BANK_ACCOUNT,
        "DRIVERS_LICENSE": ISO_LABEL_DRIVERS_LICENSE,
        "EMAIL_ADDRESS": ISO_LABEL_EMAIL_ADDRESS,
        "PASSPORT": ISO_LABEL_PASSPORT,
        "PERSON_NAME": ISO_LABEL_PERSON_NAME,
        "PHONE_NUMBER": ISO_LABEL_PHONE_NUMBER,
        "PRECISE_LOCATION": ISO_LABEL_PRECISE_LOCATION,
        "SALARY": ISO_LABEL_SALARY,
        "SSN": ISO_LABEL_SSN,
        "USERNAME": ISO_LABEL_USERNAME,
        "VEHICLE_ID": ISO_LABEL_VEHICLE_ID,
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
        # "OTHER_PII": ISO_LABEL_OTHER_PII,
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
        # "AGE",
        "BUILDINGNUM": ISO_LABEL_PRECISE_LOCATION,
        "CITY": ISO_LABEL_PRECISE_LOCATION,
        "CREDITCARDNUMBER": ISO_LABEL_PCI,
        # "DATE",
        "DRIVERLICENSENUM": ISO_LABEL_DRIVERS_LICENSE,
        "EMAIL": ISO_LABEL_EMAIL_ADDRESS,
        # "GENDER",
        "GIVENNAME": ISO_LABEL_PERSON_NAME,
        # "IDCARDNUM": ISO_LABEL_OTHER_PII,
        "PASSPORTNUM": ISO_LABEL_PASSPORT,
        # "SEX",
        "SOCIALNUM": ISO_LABEL_SSN,
        "STREET": ISO_LABEL_PRECISE_LOCATION,
        "SURNAME": ISO_LABEL_PERSON_NAME,
        # "TAXNUM",
        "TELEPHONENUM": ISO_LABEL_PHONE_NUMBER,
        # "TITLE",
        "ZIPCODE": ISO_LABEL_PRECISE_LOCATION,
    },
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
    "privacy-filter": {
        "account_number": ISO_LABEL_BANK_ACCOUNT,
        "private_address": ISO_LABEL_PRECISE_LOCATION,
        "private_email": ISO_LABEL_EMAIL_ADDRESS,
        "private_person": ISO_LABEL_PERSON_NAME,
        "private_phone": ISO_LABEL_PHONE_NUMBER,
        # "private_url": ISO_LABEL_OTHER_PII,
        # "private_date": ISO_LABEL_OTHER_PII,
        # "secret": ISO_LABEL_OTHER_PII,
    },
    r"eu-pii-safeguard*": {
        "ACCOUNT_NUMBER": ISO_LABEL_BANK_ACCOUNT,
        "ADDRESS": ISO_LABEL_PRECISE_LOCATION,
        # "AGE",
        # "AMOUNT",
        "BUILDING_NUMBER": ISO_LABEL_PRECISE_LOCATION,
        "CITY": ISO_LABEL_PRECISE_LOCATION,
        # "COMPANY_NAME",
        "COUNTRY": ISO_LABEL_PRECISE_LOCATION,
        # "CURRENCY",
        "CREDIT_CARD_NUMBER": ISO_LABEL_PCI,
        # "DEVICE_ID": ISO_LABEL_OTHER_PII,
        # "DOB": ISO_LABEL_OTHER_PII,
        "DRIVER_LICENSE": ISO_LABEL_DRIVERS_LICENSE,
        "EMAIL": ISO_LABEL_EMAIL_ADDRESS,
        # "ETHNICITY",
        "FIRSTNAME": ISO_LABEL_PERSON_NAME,
        # "GENDER",
        # "HEALTH_CONDITION",
        "HEALTH_INSURANCE_ID": ISO_LABEL_SSN,
        "IBAN": ISO_LABEL_BANK_ACCOUNT,
        "IP_ADDRESS": ISO_LABEL_IP_ADDRESS,
        # "JOB_TITLE",
        "LASTNAME": ISO_LABEL_PERSON_NAME,
        "LATITUDE": ISO_LABEL_PRECISE_LOCATION,
        "LONGITUDE": ISO_LABEL_PRECISE_LOCATION,
        # "MAC_ADDRESS": ISO_LABEL_OTHER_PII,
        "MIDDLENAME": ISO_LABEL_PERSON_NAME,
        "NATIONAL_ID": ISO_LABEL_SSN,
        "PASSPORT_NUMBER": ISO_LABEL_PASSPORT,
        # "PASSWORD": ISO_LABEL_OTHER_PII,
        "PHONE_NUMBER": ISO_LABEL_PHONE_NUMBER,
        # "POLITICAL_OPINION",
        "POSTAL_CODE": ISO_LABEL_PRECISE_LOCATION,
        # "PREFIX",
        # "RELIGION",
        "SALARY": ISO_LABEL_SALARY,
        # "SEXUAL_ORIENTATION",
        "STATE": ISO_LABEL_PRECISE_LOCATION,
        "STREET": ISO_LABEL_PRECISE_LOCATION,
        # "TAX_ID": ISO_LABEL_OTHER_PII,
        # "URL",
        "USERNAME": ISO_LABEL_USERNAME,
    },
}


def get_dataset_label_map(
    dataset_name: str, default: dict[str, str] = None
) -> dict[str, str] | None:
    for key, label_map in DATASET_LABEL_MAP.items():
        pattern = re.compile(key)
        if pattern.match(dataset_name):
            return label_map
    return default


def get_model_label_map(
    model_name: str, default: dict[str, str] = None
) -> dict[str, str] | None:
    for key, label_map in MODEL_LABEL_MAP.items():
        pattern = re.compile(key)
        if pattern.match(model_name):
            return label_map
    return default
