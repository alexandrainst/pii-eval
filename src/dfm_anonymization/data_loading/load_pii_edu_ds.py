# The Learning Agency Lab - PII Data Detection
# https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/overview

import pandas as pd
from pathlib import Path

train_data_fp = Path(
    "data/raw/kaggle/input/pii-detection-removal-from-educational-data/train.json"
)

df = pd.read_json(train_data_fp).set_index("document", inplace=False)

# train_ds = Dataset.from_dict({
#     "text": [item["text"] for item in train_data],
#     "document": [item["document"] for item in train_data],
#     "tokens": [item["tokens"] for item in train_data],
#     "trailing_whitespace": [item["trailing_whitespace"] for item in train_data],
# })

print()
