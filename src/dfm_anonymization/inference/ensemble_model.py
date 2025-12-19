from typing import List, Set, Tuple
from dfm_anonymization.inference.dataset import PiiDataEntry
from dfm_anonymization.inference.gliner_model import GlinerModel
from dfm_anonymization.inference.interfaces import PiiModel
from dfm_anonymization.inference.presidio_model import PresidioModel


class EnsembleModel(PiiModel):
    def __init__(self, name: str, models: List[Tuple[PiiModel, Set[str]]]):
        super().__init__(name)
        self.models = models

    def __call__(self, entry: PiiDataEntry) -> List[str]:
        predictions = ["O"] * len(entry.bio_tags)
        for model, label_whitelist in self.models:
            for i, new_tag in enumerate(model(
                entry=entry, label_whitelist=label_whitelist
            )):
                if predictions[i] == "O" or new_tag != "O":
                    if new_tag.startswith("I-") and \
                            predictions[i - 1][2:] != new_tag[2:]:
                        predictions[i] = f"B-{new_tag[2:]}"
                    else:
                        predictions[i] = new_tag
        return predictions

    @staticmethod
    def get_gliner_pii_presidio() -> "EnsembleModel":
        gliner_pii_labels = {
            "CARD_NUMBER", "SSN", "IBAN", "DATE_TIME", "DRIVERS_LICENSE",
            "EMAIL", "PHONE"
        }
        presidio_labels = {"URL", "PER", "ORG", "LOC"}
        return EnsembleModel(
            name="ensemble_gliner_pii_presidio",
            models=[
                (GlinerModel("nvidia/gliner-pii", gliner_pii_labels), None),
                (PresidioModel(), presidio_labels)
            ]
        )
