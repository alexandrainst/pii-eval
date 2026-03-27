from typing import List, Set, Tuple

from bidict import bidict

from dfm_anonymization.constants import *
from dfm_anonymization.dataset import PiiDataEntry
from dfm_anonymization.interfaces import PiiModel


class EnsembleModel(PiiModel):
    def __init__(self, name: str, models: List[Tuple[PiiModel, Set[str]]]):
        super().__init__(name)
        self.models = models

    def __call__(self, entry: PiiDataEntry) -> List[str]:
        n = len(entry.bio_tags) if entry.bio_tags else len(entry.tokens)
        predictions = ["O"] * n
        for model, label_whitelist in self.models:
            for i, new_tag in enumerate(
                model(entry=entry, label_whitelist=label_whitelist)
            ):
                if predictions[i] == "O" and new_tag != "O":
                    if (
                        new_tag.startswith("I-")
                        and predictions[i - 1][2:] != new_tag[2:]
                    ):
                        predictions[i] = f"B-{new_tag[2:]}"
                    else:
                        predictions[i] = new_tag
        return predictions

    @staticmethod
    def get_gliner_pii_regex() -> "EnsembleModel":
        from dfm_anonymization.models.gliner_model import GlinerModel
        from dfm_anonymization.models.regex_model import RegexModel

        return EnsembleModel(
            name="gliner_pii_and_regex",
            models=[
                (RegexModel(), ALL_ISO_LABELS),
                (
                    GlinerModel("nvidia/gliner-pii"),
                    ALL_ISO_LABELS,
                ),
            ],
        )

    @staticmethod
    def get_gliner_pii_presidio() -> "EnsembleModel":
        from dfm_anonymization.models.gliner_model import GlinerModel
        from dfm_anonymization.models.presidio_model import PresidioModel

        gliner_pii_labels = {
            ISO_LABEL_PCI,
            ISO_LABEL_SSN,
            ISO_LABEL_BANK_ACCOUNT,
            ISO_LABEL_DRIVERS_LICENSE,
            ISO_LABEL_EMAIL_ADDRESS,
            ISO_LABEL_PHONE_NUMBER,
        }
        gliner_pii_label_map = bidict(
            {k: LABEL_DESCRIPTIONS_DA[k] for k in gliner_pii_labels}
        )
        presidio_labels = {
            ISO_LABEL_PERSON_NAME,
            ISO_LABEL_PRECISE_LOCATION,
        }
        return EnsembleModel(
            name="ensemble_gliner_pii_presidio",
            models=[
                (
                    GlinerModel("nvidia/gliner-pii", labels=gliner_pii_label_map),
                    gliner_pii_labels,
                ),
                (PresidioModel(), presidio_labels),
            ],
        )
