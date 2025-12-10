from pathlib import Path
import pandas as pd

in_path = sorted(Path("data/results").glob("*.json"), reverse=True)[0]

df = pd.read_json(in_path)
means_model = df.groupby("model")[["precision", "f1", "f5", "recall"]].mean()
means_dataset = df.groupby("dataset")[["precision", "f1", "f5", "recall"]].mean()
best_models = df.loc[df.groupby("dataset")["f5"].idxmax(), ["dataset", "model", "f5"]]
for dataset, model, f5 in best_models.itertuples(index=False):
    print(f"{dataset}: {model}, f5: {f5}")

pass

# TODO: change eval script so that each label gets its own scores
