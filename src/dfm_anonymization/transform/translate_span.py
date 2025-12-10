"""
Translate a span-annotated dataset using an online LLM and update span offsets.

For every entry the script:
1. Translates the full text.
2. Translates every span independently.
3. Searches for the translated spans in the translated text to recover offsets.
If any span cannot be re-located the whole entry is skipped (as specified by the
module level comment that previously lived here).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib import error, request


LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET = "mattmdjaga"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TARGET_LANG = "Danish"
DEFAULT_SOURCE_LANG = "English"


class ChatCompletionsClient:
    """Minimal OpenAI-compatible chat completions client."""

    def __init__(
        self, model: str, api_key: str, api_base: str, timeout: float = 60.0
    ) -> None:
        if not api_key:
            raise ValueError(
                "No API key provided. Set --api-key or OPENAI_API_KEY."
            )
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def complete(self, messages: Sequence[dict], temperature: float = 0.0) -> str:
        payload = json.dumps(
            {"model": self.model, "temperature": temperature, "messages": messages}
        ).encode("utf-8")
        url = f"{self.api_base}/chat/completions"
        req = request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(
                f"LLM request failed ({exc.code}): {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        data = json.loads(raw)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response: {data}") from exc
        return content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a span dataset via an online LLM."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset name used for deriving default paths.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Path to the source dataset. Defaults to "
        "data/processed/span/reduced/{dataset}.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Where to store the translated dataset. Defaults to "
        "data/processed/span/translated/{dataset}_{target_lang}.json",
    )
    parser.add_argument(
        "--target-language",
        default=DEFAULT_TARGET_LANG,
        help="Target language for the translation.",
    )
    parser.add_argument(
        "--source-language",
        default=DEFAULT_SOURCE_LANG,
        help="Source language hint for the LLM.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model identifier (OpenAI-compatible).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="Base URL for the chat completion endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for the LLM provider.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for LLM calls (seconds).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit of samples to process for quick experiments.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _default_input_path(dataset: str) -> Path:
    return Path(f"data/processed/span/reduced/{dataset}.json")


def _default_output_path(dataset: str, target_language: str) -> Path:
    safe_target = target_language.lower().replace(" ", "_")
    return Path(f"data/processed/span/translated/{dataset}_{safe_target}.json")


def _translate_full_text(
    client: ChatCompletionsClient,
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    if not text:
        return ""
    messages = [
        {
            "role": "system",
            "content": "You precisely translate text while preserving structure.",
        },
        {
            "role": "user",
            "content": (
                f"Translate the following {source_language} text into "
                f"{target_language}. Preserve formatting and do not add"
                " commentary.\n"
                "----\n"
                f"{text}"
            ),
        },
    ]
    return client.complete(messages)


def _translate_spans(
    client: ChatCompletionsClient,
    spans: Sequence[str],
    source_language: str,
    target_language: str,
) -> List[str]:
    if not spans:
        return []
    payload = json.dumps(list(spans), ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": (
                "You translate spans and respond with pure JSON arrays."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate each of the following strings from {source_language} "
                f"to {target_language}. Respond with a JSON array of the "
                "translated strings in the same order with no commentary.\n"
                f"{payload}"
            ),
        },
    ]
    raw = client.complete(messages)
    extracted = _extract_json_array(raw)
    if len(extracted) != len(spans):
        raise RuntimeError(
            "Mismatch between requested span translations and received output."
        )
    return [str(item) for item in extracted]


def _extract_json_array(raw: str) -> List[str]:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"Expected JSON array in LLM response: {raw}")
    snippet = raw[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed parsing span translation JSON: {raw}") from exc
    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected list response, got: {parsed!r}")
    return parsed


def _resolve_span_text(span: dict, full_text: str) -> str:
    text = span.get("text")
    if text:
        return text
    start = int(span.get("start", 0))
    end = int(span.get("end", start))
    return full_text[start:end]


def _align_spans(
    original_text: str,
    translated_text: str,
    spans: Sequence[dict],
    translated_spans: Sequence[str],
) -> list[dict] | None:
    aligned: list[dict] = []
    cursor = 0
    original_len = max(1, len(original_text))
    translated_len = len(translated_text)
    for span, translated in zip(spans, translated_spans):
        translated = translated.strip()
        if not translated:
            LOGGER.debug("Empty translation for span %s", span)
            return None
        approx_ratio = max(0, int(span.get("start", 0))) / original_len
        approx_idx = int(approx_ratio * translated_len)
        start_idx, end_idx = _locate_span(
            translated_text, translated, max(cursor, approx_idx)
        )
        if start_idx is None:
            LOGGER.debug(
                "Failed to match translated span '%s' in document %s",
                translated,
                span.get("label"),
            )
            return None
        cursor = end_idx
        aligned.append(
            {
                "label": span.get("label"),
                "start": start_idx,
                "end": end_idx,
                "text": translated_text[start_idx:end_idx],
            }
        )
    return aligned


def _locate_span(text: str, needle: str, hint_idx: int) -> tuple[int, int] | tuple[None, None]:
    def _search_ranges() -> Iterable[int]:
        window = max(0, hint_idx - 512)
        yield window
        yield 0

    for start in _search_ranges():
        idx = text.find(needle, start)
        if idx != -1:
            return idx, idx + len(needle)
    needle_lower = needle.lower()
    text_lower = text.lower()
    for start in _search_ranges():
        idx = text_lower.find(needle_lower, start)
        if idx != -1:
            return idx, idx + len(needle)
    return (None, None)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    input_path = args.input_path or _default_input_path(args.dataset)
    output_path = args.output_path or _default_output_path(
        args.dataset, args.target_language
    )
    LOGGER.info("Reading dataset from %s", input_path)
    samples = json.loads(input_path.read_text(encoding="utf-8"))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    LOGGER.info("Loaded %d samples", len(samples))
    client = ChatCompletionsClient(
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        timeout=args.timeout,
    )
    translated_samples = []
    skipped = 0
    for entry in samples:
        try:
            translated_text = _translate_full_text(
                client,
                entry.get("full_text", ""),
                args.source_language,
                args.target_language,
            )
            spans = entry.get("spans", [])
            span_texts = [_resolve_span_text(span, entry.get("full_text", "")) for span in spans]
            translated_span_texts = _translate_spans(
                client,
                span_texts,
                args.source_language,
                args.target_language,
            )
            aligned_spans = _align_spans(
                entry.get("full_text", ""),
                translated_text,
                spans,
                translated_span_texts,
            )
            if aligned_spans is None:
                skipped += 1
                continue
            translated_samples.append(
                {
                    "document": entry.get("document"),
                    "full_text": translated_text,
                    "spans": aligned_spans,
                }
            )
        except Exception as exc:
            skipped += 1
            LOGGER.warning("Skipping entry %s due to error: %s", entry.get("document"), exc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(translated_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "Wrote %d translated samples to %s (%d skipped)",
        len(translated_samples),
        output_path,
        skipped,
    )


if __name__ == "__main__":
    main()

