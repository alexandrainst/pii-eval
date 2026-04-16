from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk

from dfm_anonymization.dataset import PiiDataset
from dfm_anonymization.util.spans import LabelSpan

DS_DA_CACHE_PATH = Path("data/raw/openpii-1m-da")
PII_DS_CACHE_PATH = Path("data/ground_truth/openpii-1m-da.json")


def save_danish_part_to_disk():
    ds = load_dataset("ai4privacy/pii-masking-openpii-1m")
    # Filter to Danish entries and merge train and validation sets
    ds_train_da = ds["train"].filter(lambda x: x["language"] == "da")
    ds_val_da = ds["validation"].filter(lambda x: x["language"] == "da")
    ds_da = concatenate_datasets([ds_train_da, ds_val_da])
    ds_da.save_to_disk(DS_DA_CACHE_PATH)


if not DS_DA_CACHE_PATH.exists():
    save_danish_part_to_disk()

if not PII_DS_CACHE_PATH.exists():
    ds = load_from_disk(DS_DA_CACHE_PATH)
    pii_ds = PiiDataset.from_span_dataset(
        dataset=[
            (
                x["source_text"],
                [
                    LabelSpan(start=y["start"], end=y["end"], label=y["label"])
                    for y in x["privacy_mask"]
                ],
            )
            for x in ds
        ]
    )
    pii_ds.to_file(PII_DS_CACHE_PATH)

pass
