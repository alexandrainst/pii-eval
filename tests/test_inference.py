import pytest

from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.inference import inference
from dfm_anonymization.interfaces import PiiModel


class _FailingModel(PiiModel):
    def __init__(self) -> None:
        super().__init__(name="failing-model")

    def __call__(self, entry: PiiDataEntry, label_whitelist=None) -> list[str]:
        raise ValueError("backend exploded")


def test_inference_raises_contextual_error_when_model_fails() -> None:
    dataset = PiiDataset(
        entries=[PiiDataEntry(id=42, text="Alice", tokens=["Alice"], bio_tags=["O"])],
        metadata={"dataset_name": "sample"},
    )

    with pytest.raises(RuntimeError, match="Inference failed for model 'failing-model' on entry '42'.") as exc_info:
        inference(_FailingModel(), dataset)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "backend exploded"