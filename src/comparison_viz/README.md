# comparison_viz

A small web app for comparing ground-truth BIO tags against a model's predictions, visualized
side-by-side over the source text.

## What it does

- Reads ground truth from `data/ground_truth/<dataset>.json`
- Reads predictions from `data/predictions/<model>/<dataset>.json`
- Renders the same text twice (GT on the left, prediction on the right) with each BIO span
  highlighted in a color derived from its **canonical ISO label** (using `MODEL_LABEL_MAP` and
  `DATASET_LABEL_MAP` from `dfm_anonymization.constants`). The original raw label is shown as a
  small superscript and in the hover tooltip.
- Computes per-entry agreement metrics (exact span match + token-level over canonical labels).

## Run

```
uv run python -m comparison_viz
# or
uv run python -m comparison_viz --host 0.0.0.0 --port 8001 --data-dir data
```

Then open http://127.0.0.1:8000 in a browser.

## Controls

- Dataset / model / entry dropdowns
- `«` / `»` buttons or the `←` / `→` arrow keys to step through entries
- Hover any highlighted span for details (ISO label, raw label, token range, char range)

## Layout

```
src/comparison_viz/
├── __init__.py
├── __main__.py        # CLI entrypoint
├── server.py          # FastAPI app + endpoints
├── data_loader.py     # discovery + JSON loading
├── span_utils.py      # BIO -> spans, char-offset reconstruction, label map, metrics
└── static/
    ├── index.html
    ├── app.js
    └── styles.css
```

## API

- `GET /api/datasets` -> `["contracts", "cvs", ...]`
- `GET /api/models` -> `["eu-pii-safeguard", ...]`
- `GET /api/entries?dataset=cvs&model=eu-pii-safeguard`
- `GET /api/entry?dataset=cvs&id=<entry_id>&model=eu-pii-safeguard`
