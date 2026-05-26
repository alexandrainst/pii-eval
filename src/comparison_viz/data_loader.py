"""Discovery + loading of dataset and prediction JSON files."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


SKIP_DIRS = {"disabled"}


class DataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.gt_dir = self.data_dir / "ground_truth/dimmi"
        self.pred_dir = self.data_dir / "predictions_dimmi"

    # --- discovery ----------------------------------------------------------
    def list_datasets(self) -> list[str]:
        if not self.gt_dir.is_dir():
            return []
        return sorted(
            p.stem
            for p in self.gt_dir.glob("*.json")
            if p.is_file()
            and p.parent.name not in SKIP_DIRS
            and not p.name.startswith(".")
        )

    def list_models(self) -> list[str]:
        if not self.pred_dir.is_dir():
            return []
        return sorted(
            p.name
            for p in self.pred_dir.iterdir()
            if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith(".")
        )

    # --- loading ------------------------------------------------------------
    @lru_cache(maxsize=64)
    def load_ground_truth(self, dataset: str) -> dict:
        path = self.gt_dir / f"{dataset}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No ground truth for dataset '{dataset}': {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @lru_cache(maxsize=128)
    def load_prediction(self, model: str, dataset: str) -> dict:
        path = self.pred_dir / model / f"{dataset}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"No prediction for model '{model}' / dataset '{dataset}': {path}"
            )
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_entry_ids(self, dataset: str) -> list[str]:
        gt = self.load_ground_truth(dataset)
        return [e["id"] for e in gt.get("entries", [])]

    def get_entry(self, dataset: str, entry_id: str) -> dict | None:
        for e in self.load_ground_truth(dataset).get("entries", []):
            if e.get("id") == entry_id:
                return e
        return None

    def get_prediction_entry(
        self, model: str, dataset: str, entry_id: str
    ) -> dict | None:
        try:
            pred = self.load_prediction(model, dataset)
        except FileNotFoundError:
            return None
        for e in pred.get("entries", []):
            if e.get("id") == entry_id:
                return e
        return None
