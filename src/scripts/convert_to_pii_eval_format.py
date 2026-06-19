from pathlib import Path

from tqdm import tqdm

from pii_eval.data import PiiDataset


files = list(Path("data/ground_truth").glob("*.json"))

for path in tqdm(files, dynamic_ncols=True):
    ds = PiiDataset.load_old_format(path)
    ds.save_json(f"data/ground_truth_new/{ds.name}.json")
