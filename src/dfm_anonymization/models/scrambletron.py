from dfm_anonymization.models.presidio_model import PresidioModel
from dfm_anonymization.util.util import get_torch_device

from scrambletron.main import create_analyzer


class ScrambletronModel(PresidioModel):
    def __init__(
        self,
        name: str = "scrambletron",
    ):
        super().__init__(name)
        device = get_torch_device()
        self.analyzer = create_analyzer(device=device)
