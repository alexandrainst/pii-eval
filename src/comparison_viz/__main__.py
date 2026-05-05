"""CLI entrypoint: ``python -m comparison_viz``."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Web visualizer for ground-truth vs prediction BIO tags."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the data/ directory (containing ground_truth/ and predictions/).",
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    app = create_app(data_dir)
    print(f"Serving from data dir: {data_dir}")
    print(f"Open http://{args.host}:{args.port} in your browser.")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
