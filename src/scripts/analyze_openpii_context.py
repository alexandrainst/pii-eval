"""Analyze context surrounding PII spans in openpii-1m-da-s dataset."""

import json
from pathlib import Path


def extract_spans(entry):
    """Extract labeled spans from BIO-tagged tokens using token_spans for positions."""
    tokens = entry["tokens"]
    bio_tags = entry["bio_tags"]
    text = entry["text"]
    token_spans = entry["token_spans"]

    spans = []
    current_label = None
    current_start = None
    current_end = None

    for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
        char_start, char_end = token_spans[i]
        if tag.startswith("B-"):
            if current_label is not None:
                spans.append((current_label, current_start, current_end))
            current_label = tag[2:]
            current_start = char_start
            current_end = char_end
        elif tag.startswith("I-") and current_label:
            current_end = char_end
        else:
            if current_label is not None:
                spans.append((current_label, current_start, current_end))
            current_label = None

    if current_label is not None:
        spans.append((current_label, current_start, current_end))

    results = []
    for label, start, end in spans:
        span_text = text[start:end]
        context_before = text[max(0, start - 80):start]
        context_after = text[end:end + 40]
        results.append((label, span_text, context_before, context_after))

    return results


def luhn_check(card_number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13:
        return False
    digits.reverse()
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def main():
    data_path = Path("data/ground_truth/openpii-1m-da-s.json")
    with open(data_path) as f:
        raw = json.load(f)

    data = raw["entries"]
    print(f"Loaded {len(data)} entries\n")

    # Collect spans by label
    spans_by_label = {}
    for entry in data:
        for label, span_text, ctx_before, ctx_after in extract_spans(entry):
            if label not in spans_by_label:
                spans_by_label[label] = []
            spans_by_label[label].append((span_text, ctx_before, ctx_after))

    # Print label distribution
    print("=== LABEL DISTRIBUTION ===")
    for label, spans in sorted(spans_by_label.items(), key=lambda x: -len(x[1])):
        print(f"  {label}: {len(spans)}")
    print()

    # --- DRIVERLICENSENUM ---
    label = "DRIVERLICENSENUM"
    spans = spans_by_label.get(label, [])
    print(f"=== {label} ({len(spans)} total) — showing up to 40 ===")
    for i, (span_text, ctx_before, ctx_after) in enumerate(spans[:40]):
        print(f"\n  [{i+1}] SPAN: {span_text!r}")
        print(f"       BEFORE: ...{ctx_before!r}")
        print(f"       AFTER:  {ctx_after!r}")
    print()

    # --- PASSPORTNUM ---
    label = "PASSPORTNUM"
    spans = spans_by_label.get(label, [])
    print(f"=== {label} ({len(spans)} total) — showing up to 30 ===")
    for i, (span_text, ctx_before, ctx_after) in enumerate(spans[:30]):
        print(f"\n  [{i+1}] SPAN: {span_text!r}")
        print(f"       BEFORE: ...{ctx_before!r}")
        print(f"       AFTER:  {ctx_after!r}")
    print()

    # --- CREDITCARDNUMBER ---
    label = "CREDITCARDNUMBER"
    spans = spans_by_label.get(label, [])
    print(f"=== {label} ({len(spans)} total) — showing up to 30 ===")
    luhn_pass = 0
    luhn_fail = 0
    for i, (span_text, ctx_before, ctx_after) in enumerate(spans[:30]):
        passes = luhn_check(span_text)
        tag = "LUHN-OK" if passes else "LUHN-FAIL"
        print(f"\n  [{i+1}] SPAN: {span_text!r}  [{tag}]")
        print(f"       BEFORE: ...{ctx_before!r}")
        print(f"       AFTER:  {ctx_after!r}")

    # Check all for Luhn
    for span_text, _, _ in spans:
        if luhn_check(span_text):
            luhn_pass += 1
        else:
            luhn_fail += 1
    print(f"\n  LUHN SUMMARY: {luhn_pass} pass, {luhn_fail} fail out of {len(spans)} total")
    print()

    # --- IDCARDNUM ---
    label = "IDCARDNUM"
    spans = spans_by_label.get(label, [])
    print(f"=== {label} ({len(spans)} total) — showing up to 20 ===")
    for i, (span_text, ctx_before, ctx_after) in enumerate(spans[:20]):
        print(f"\n  [{i+1}] SPAN: {span_text!r}")
        print(f"       BEFORE: ...{ctx_before!r}")
        print(f"       AFTER:  {ctx_after!r}")
    print()

    # --- STREET ---
    label = "STREET"
    spans = spans_by_label.get(label, [])
    print(f"=== {label} ({len(spans)} total) — showing up to 30 ===")
    for i, (span_text, ctx_before, ctx_after) in enumerate(spans[:30]):
        print(f"\n  [{i+1}] SPAN: {span_text!r}")
        print(f"       BEFORE: ...{ctx_before!r}")
        print(f"       AFTER:  {ctx_after!r}")
    print()


if __name__ == "__main__":
    main()
