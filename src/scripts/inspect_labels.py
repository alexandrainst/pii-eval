
from dfm_anonymization.dataset import PiiDataset

ds = PiiDataset.from_file("data/ground_truth/openpii-1m-da-s.json")

print(ds.labels())

