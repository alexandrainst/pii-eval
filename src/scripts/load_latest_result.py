import json
import pandas as pd
from pathlib import Path

x = [
    pd.DataFrame(json.loads(x.read_text()))
    for x in sorted(list(Path("data/results").glob("*.json")), reverse=True)
]


pass
