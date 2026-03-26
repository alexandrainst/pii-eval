from dfm_anonymization.dataset import PiiDataEntry, PiiDataset
from dfm_anonymization.evaluate import evaluate


def _dataset(entries: list[PiiDataEntry], dataset_name: str, model_name: str | None = None) -> PiiDataset:
    metadata = {"dataset_name": dataset_name}
    if model_name is not None:
        metadata["model_name"] = model_name
    return PiiDataset(entries=entries, metadata=metadata)


def test_evaluate_aligns_predictions_by_entry_id_for_shuffled_subset() -> None:
    ground_truth = _dataset(
        [
            PiiDataEntry(id=0, text="A", tokens=["A"], bio_tags=["O"]),
            PiiDataEntry(
                id=1090,
                text="Call 44558899",
                tokens=["Call", "44558899"],
                bio_tags=["O", "B-PHONE_NUMBER"],
            ),
            PiiDataEntry(id=2, text="B", tokens=["B"], bio_tags=["O"]),
        ],
        dataset_name="gen_presidio",
    )
    predictions = _dataset(
        [
            PiiDataEntry(id=2, text="B", tokens=["B"], bio_tags=["O"]),
            PiiDataEntry(
                id=1090,
                text="Call 44558899",
                tokens=["Call", "44558899"],
                bio_tags=["O", "B-PHONE_NUMBER"],
            ),
        ],
        dataset_name="gen_presidio",
        model_name="gemma-3n-e4b-it",
    )

    result = evaluate(ground_truth, predictions)["PHONE_NUMBER"]

    assert result.tp == 1
    assert result.fp == 0
    assert result.fn == 0


def test_evaluate_rejects_mismatched_tag_lengths() -> None:
    ground_truth = _dataset(
        [
            PiiDataEntry(
                id=1,
                text="Call 44558899",
                tokens=["Call", "44558899"],
                bio_tags=["O", "B-PHONE_NUMBER"],
            )
        ],
        dataset_name="gen_presidio",
    )
    predictions = _dataset(
        [
            PiiDataEntry(
                id=1,
                text="Call 44558899",
                tokens=["Call", "44558899"],
                bio_tags=["O"],
            )
        ],
        dataset_name="gen_presidio",
        model_name="gemma-3n-e4b-it",
    )

    try:
        evaluate(ground_truth, predictions)
    except ValueError as exc:
        assert "Tag length mismatch" in str(exc)
    else:
        raise AssertionError("Expected evaluate() to reject mismatched tag lengths")
