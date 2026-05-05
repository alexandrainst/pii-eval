"""FastAPI server exposing endpoints for the comparison visualization."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .data_loader import DataLoader
from .span_utils import (
    bio_to_spans,
    compute_metrics,
    map_labels_for_dataset,
    map_labels_for_model,
    reconstruct_token_spans,
    spans_to_dicts,
)


def create_app(data_dir: Path) -> FastAPI:
    data_dir = Path(data_dir).resolve()
    loader = DataLoader(data_dir)
    static_dir = Path(__file__).parent / "static"

    app = FastAPI(title="BIO Comparison Visualizer", docs_url="/api/docs")

    @app.get("/api/datasets")
    def datasets():
        return loader.list_datasets()

    @app.get("/api/models")
    def models():
        return loader.list_models()

    @app.get("/api/entries")
    def entries(dataset: str = Query(...), model: str | None = Query(None)):
        try:
            gt = loader.load_ground_truth(dataset)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

        pred_ids: set[str] = set()
        if model:
            try:
                pred = loader.load_prediction(model, dataset)
                pred_ids = {e["id"] for e in pred.get("entries", [])}
            except FileNotFoundError:
                pred_ids = set()

        result = []
        for e in gt.get("entries", []):
            eid = e["id"]
            n_gt_spans = sum(1 for t in e.get("bio_tags", []) if t.startswith("B-"))
            result.append(
                {
                    "id": eid,
                    "gt_span_count": n_gt_spans,
                    "has_prediction": eid in pred_ids,
                }
            )
        return result

    @app.get("/api/entry")
    def entry(
        dataset: str = Query(...),
        id: str = Query(...),
        model: str | None = Query(None),
    ):
        gt_entry = loader.get_entry(dataset, id)
        if gt_entry is None:
            raise HTTPException(404, f"Entry '{id}' not found in dataset '{dataset}'")

        text: str = gt_entry["text"]
        tokens: list[str] = gt_entry["tokens"]
        gt_bio: list[str] = gt_entry.get("bio_tags", [])

        # Try to use prediction's token_spans (they cover the same tokens),
        # otherwise reconstruct from text.
        pred_entry = loader.get_prediction_entry(model, dataset, id) if model else None

        token_spans = None
        if pred_entry and pred_entry.get("token_spans"):
            ts = pred_entry["token_spans"]
            if len(ts) == len(tokens):
                token_spans = [tuple(s) for s in ts]
        if token_spans is None:
            token_spans = reconstruct_token_spans(text, tokens)

        ds_map = map_labels_for_dataset(dataset)
        gt_spans = bio_to_spans(gt_bio, tokens, token_spans, text, ds_map)

        pred_payload = None
        metrics = None
        if pred_entry is not None:
            pred_bio = pred_entry.get("bio_tags", [])
            mdl_map = map_labels_for_model(model) if model else {}
            pred_spans = bio_to_spans(pred_bio, tokens, token_spans, text, mdl_map)
            pred_payload = {
                "bio_tags": pred_bio,
                "spans": spans_to_dicts(pred_spans),
            }
            metrics = compute_metrics(gt_spans, pred_spans, len(tokens))

        return JSONResponse(
            {
                "id": id,
                "dataset": dataset,
                "model": model,
                "text": text,
                "tokens": tokens,
                "token_spans": [list(s) for s in token_spans],
                "ground_truth": {
                    "bio_tags": gt_bio,
                    "spans": spans_to_dicts(gt_spans),
                },
                "prediction": pred_payload,
                "metrics": metrics,
            }
        )

    # --- static frontend ----------------------------------------------------
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(static_dir / "index.html"))

    return app
