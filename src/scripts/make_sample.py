

import random

from dfm_anonymization.dataset import PiiDataset


big_ds = PiiDataset.from_file("data/ground_truth/openpii-1m-da.json")

sample = PiiDataset(metadata=big_ds.metadata, entries=random.sample(big_ds.entries, 5000))

sample.to_file("data/ground_truth/openpii-1m-da-s.json")
