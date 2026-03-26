import json
from pathlib import Path

from scripts.add_label_list_to_metadata import update_dataset_metadata


def test_update_dataset_metadata_adds_unique_labels_to_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sample.json"
    dataset_path.write_text(
        json.dumps(
            {
                "metadata": {"dataset_name": "sample"},
                "entries": [
                    {"bio_tags": ["O", "B-PRECISE_LOCATION", "I-PRECISE_LOCATION"]},
                    {"bio_tags": ["B-PERSON", "I-PERSON", "O", "B-PRECISE_LOCATION"]},
                ],
            }
        ),
        encoding="utf8",
    )

    labels = update_dataset_metadata(dataset_path)
    updated = json.loads(dataset_path.read_text(encoding="utf8"))

    assert labels == ["PERSON", "PRECISE_LOCATION"]
    assert updated["metadata"] == {
        "dataset_name": "sample",
        "labels": ["PERSON", "PRECISE_LOCATION"],
    }
