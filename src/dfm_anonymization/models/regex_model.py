from ..constants import *
from ..dataset import PiiDataEntry
from ..interfaces import PiiModel

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


class RegexModel(PiiModel):
    def __init__(
        self,
        name: str = "regex",
    ):
        super().__init__(name)

    def __call__(
        self, entry: PiiDataEntry, label_whitelist: set[str] | None = None
    ) -> list[str]:

        # TODO:
        # This model uses regex find PII information in text from the incoming
        # dataset-entry. It only works with the above regex-compatible labels,
        # and some of them only partially - for example for the precise location
        # label we will focus on catching coordinates (easier for regex than
        # the multitude of ways to format addresses containing house numbers)



        pass
