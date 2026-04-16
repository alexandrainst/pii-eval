"""Analyze false negatives and false positives for specific PII types in regex predictions."""

import json
from collections import defaultdict


def load_data(gt_path, pred_path):
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)
    return gt, pred


def extract_spans(tokens, bio_tags, target_labels):
    """Extract spans of consecutive tokens for target B/I labels.
    target_labels: set of base labels (without B-/I- prefix).
    Returns list of (start_idx, end_idx, label, span_text)."""
    spans = []
    i = 0
    while i < len(bio_tags):
        tag = bio_tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            if label in target_labels:
                start = i
                end = i
                while end + 1 < len(bio_tags) and bio_tags[end + 1] == f"I-{label}":
                    end += 1
                span_text = " ".join(tokens[start:end + 1])
                spans.append((start, end, label, span_text))
                i = end + 1
                continue
        i += 1
    return spans


def get_context(text, tokens, token_spans, start_idx, n_chars=100):
    """Get preceding context from entry text."""
    if token_spans:
        char_start = token_spans[start_idx][0]
        ctx_start = max(0, char_start - n_chars)
        return text[ctx_start:char_start].strip()
    else:
        # Fallback: reconstruct from tokens
        preceding = " ".join(tokens[max(0, start_idx - 15):start_idx])
        return preceding[-n_chars:]


def find_false_negatives(gt_data, pred_data, gt_label, pred_label):
    """Find GT spans that are not covered by prediction spans."""
    gt_entries = {e["id"]: e for e in gt_data["entries"]}
    pred_entries = {e["id"]: e for e in pred_data["entries"]}

    fn_examples = []

    for eid, gt_entry in gt_entries.items():
        pred_entry = pred_entries.get(eid)
        if pred_entry is None:
            continue

        gt_tags = gt_entry["bio_tags"]
        pred_tags = pred_entry["bio_tags"]
        tokens = gt_entry["tokens"]
        text = gt_entry.get("text", "")
        token_spans = gt_entry.get("token_spans")

        # Extract GT spans for the target label
        gt_spans = extract_spans(tokens, gt_tags, {gt_label})

        for start, end, label, span_text in gt_spans:
            # Check if prediction covers this span
            is_missed = True
            for idx in range(start, end + 1):
                if idx < len(pred_tags):
                    ptag = pred_tags[idx]
                    if ptag in (f"B-{pred_label}", f"I-{pred_label}"):
                        is_missed = False
                        break

            if is_missed:
                context = get_context(text, tokens, token_spans, start)
                fn_examples.append({
                    "id": eid,
                    "span": span_text,
                    "context": context,
                    "pred_tags_at_span": [pred_tags[i] if i < len(pred_tags) else "?" for i in range(start, end + 1)],
                })

    return fn_examples


def find_false_positives(gt_data, pred_data, gt_label, pred_label):
    """Find prediction spans that are not in GT."""
    gt_entries = {e["id"]: e for e in gt_data["entries"]}
    pred_entries = {e["id"]: e for e in pred_data["entries"]}

    fp_examples = []

    for eid, pred_entry in pred_entries.items():
        gt_entry = gt_entries.get(eid)
        if gt_entry is None:
            continue

        gt_tags = gt_entry["bio_tags"]
        pred_tags = pred_entry["bio_tags"]
        tokens = pred_entry["tokens"]
        text = pred_entry.get("text", "")
        token_spans = gt_entry.get("token_spans")

        # Extract predicted spans
        pred_spans = extract_spans(tokens, pred_tags, {pred_label})

        for start, end, label, span_text in pred_spans:
            # Check if GT has the corresponding label
            is_fp = True
            for idx in range(start, end + 1):
                if idx < len(gt_tags):
                    gtag = gt_tags[idx]
                    if gtag in (f"B-{gt_label}", f"I-{gt_label}"):
                        is_fp = False
                        break

            if is_fp:
                context = get_context(text, tokens, token_spans, start)
                gt_tags_at_span = [gt_tags[i] if i < len(gt_tags) else "?" for i in range(start, end + 1)]
                fp_examples.append({
                    "id": eid,
                    "span": span_text,
                    "context": context,
                    "gt_tags_at_span": gt_tags_at_span,
                })

    return fp_examples


def print_examples(examples, title, max_examples, is_fp=False):
    print(f"\n{'='*120}")
    print(f" {title} — {len(examples)} total, showing {min(max_examples, len(examples))}")
    print(f"{'='*120}")
    for i, ex in enumerate(examples[:max_examples]):
        tags_key = "gt_tags_at_span" if is_fp else "pred_tags_at_span"
        tags_str = ", ".join(ex[tags_key])
        print(f"\n--- Example {i+1} (entry {ex['id']}) ---")
        print(f"  Span:    [{ex['span']}]")
        print(f"  Tags:    {tags_str}")
        print(f"  Context: ...{ex['context']}|>>HERE<<|")


def main():
    gt_path = "data/ground_truth/openpii-1m-da-s.json"
    pred_path = "data/predictions/regex/openpii-1m-da-s.json"

    print("Loading data...")
    gt_data, pred_data = load_data(gt_path, pred_path)
    print(f"  GT entries: {len(gt_data['entries'])}")
    print(f"  Pred entries: {len(pred_data['entries'])}")

    # --- FALSE NEGATIVES ---

    # 1. DRIVERLICENSENUM (GT) -> DRIVERS_LICENSE (pred)
    fn_dl = find_false_negatives(gt_data, pred_data, "DRIVERLICENSENUM", "DRIVERS_LICENSE")
    print_examples(fn_dl, "FALSE NEGATIVES: DRIVERLICENSENUM (missed by regex)", 50)

    # 2. PASSPORTNUM (GT) -> PASSPORT (pred)
    fn_pp = find_false_negatives(gt_data, pred_data, "PASSPORTNUM", "PASSPORT")
    print_examples(fn_pp, "FALSE NEGATIVES: PASSPORTNUM (missed by regex)", 30)

    # 3. CREDITCARDNUMBER (GT) -> PCI (pred)
    fn_cc = find_false_negatives(gt_data, pred_data, "CREDITCARDNUMBER", "PCI")
    print_examples(fn_cc, "FALSE NEGATIVES: CREDITCARDNUMBER (missed by regex)", 30)

    # --- FALSE POSITIVES ---

    # DRIVERS_LICENSE FP
    fp_dl = find_false_positives(gt_data, pred_data, "DRIVERLICENSENUM", "DRIVERS_LICENSE")
    print_examples(fp_dl, "FALSE POSITIVES: DRIVERS_LICENSE (falsely predicted)", 20, is_fp=True)


if __name__ == "__main__":
    main()
