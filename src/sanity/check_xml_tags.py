import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


XML_TAG_PATTERN = re.compile(
	r'<([A-Z_]+)\s+text\s*=\s*(?:"([^"]*)"|\\"([^"]*)\\")\s*/?>'
)


@dataclass
class XmlEntity:
	category: str
	text: str
	start: int
	end: int


def parse_xml_example(example: str) -> tuple[str, list[XmlEntity]]:
	entities: list[XmlEntity] = []
	clean_text = example
	offset = 0

	for match in XML_TAG_PATTERN.finditer(example):
		category = match.group(1)
		entity_text = match.group(2) if match.group(2) is not None else match.group(3)

		start = match.start() - offset
		end = start + len(entity_text)
		entities.append(
			XmlEntity(category=category, text=entity_text, start=start, end=end)
		)

		clean_text = (
			clean_text[: match.start() - offset]
			+ entity_text
			+ clean_text[match.end() - offset :]
		)
		offset += len(match.group(0)) - len(entity_text)

		# Remove one matching closing tag if the XML was provided in paired form:
		# <TAG text="..."> </TAG>
		closing_tag = f"</{category}>"
		closing_pos = clean_text.find(closing_tag, end)
		if closing_pos != -1:
			clean_text = (
				clean_text[:closing_pos]
				+ clean_text[closing_pos + len(closing_tag) :]
			)
			offset += len(closing_tag)

	return clean_text, entities


def build_token_positions(text: str, tokens: list[str]) -> list[tuple[int, int]]:
	positions: list[tuple[int, int]] = []
	search_start = 0

	for token in tokens:
		token_start = text.find(token, search_start)
		if token_start == -1:
			token_start = search_start
			token_end = search_start
		else:
			token_end = token_start + len(token)
			search_start = token_end
		positions.append((token_start, token_end))

	return positions


def get_overlapping_token_indices(
	token_positions: list[tuple[int, int]],
	start: int,
	end: int,
) -> list[int]:
	overlapping: list[int] = []
	for idx, (tok_start, tok_end) in enumerate(token_positions):
		if tok_start < end and tok_end > start:
			overlapping.append(idx)
	return overlapping


def entity_has_expected_bio(
	bio_tags: list[str],
	token_indices: list[int],
	category: str,
) -> bool:
	if not token_indices:
		return False

	expected_b = f"B-{category}"
	expected_i = f"I-{category}"

	first_tag = bio_tags[token_indices[0]]
	if first_tag != expected_b:
		return False

	for idx in token_indices[1:]:
		if bio_tags[idx] != expected_i:
			return False

	return True


def tokenize_text(text: str) -> list[str]:
	tokens: list[str] = []
	for word in text.split():
		tokens.extend(re.findall(r"\w+|[^\w\s]", word))
	return tokens


def find_subsequence_ranges(tokens: list[str], pattern: list[str]) -> list[tuple[int, int]]:
	if not pattern:
		return []

	ranges: list[tuple[int, int]] = []
	window = len(pattern)
	max_start = len(tokens) - window
	for start in range(max_start + 1):
		if tokens[start : start + window] == pattern:
			ranges.append((start, start + window))
	return ranges


def entity_matches_bio_by_tokens(
	tokens: list[str],
	bio_tags: list[str],
	category: str,
	entity_text: str,
) -> bool:
	entity_tokens = tokenize_text(entity_text)
	if not entity_tokens:
		return False

	for start, end in find_subsequence_ranges(tokens, entity_tokens):
		indices = list(range(start, end))
		if entity_has_expected_bio(bio_tags, indices, category):
			return True

	return False


def build_context_snippet(text: str, needle: str, radius: int = 40) -> str:
	if not needle:
		return text[: radius * 2]

	position = text.find(needle)
	if position == -1:
		return text[: radius * 2]

	start = max(0, position - radius)
	end = min(len(text), position + len(needle) + radius)
	prefix = "..." if start > 0 else ""
	suffix = "..." if end < len(text) else ""
	return f"{prefix}{text[start:end]}{suffix}"


def load_yaml_examples(yaml_dir: Path) -> list[tuple[str, list[XmlEntity]]]:
	examples: list[tuple[str, list[XmlEntity]]] = []

	for yaml_path in sorted(yaml_dir.glob("*.yaml")):
		with yaml_path.open("r", encoding="utf-8") as handle:
			payload = yaml.safe_load(handle) or {}

		response_content = payload.get("response_content", "")
		if not response_content:
			continue

		parts = [part.strip() for part in response_content.split("\n\n") if part.strip()]
		for part in parts:
			clean_text, entities = parse_xml_example(part)
			examples.append((clean_text, entities))

	return examples


def load_gen_prompt_entries(gen_prompt_path: Path) -> list[dict]:
	with gen_prompt_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	entries = payload.get("entries", [])
	valid_entries = [
		entry
		for entry in entries
		if entry.get("text") is not None
		and isinstance(entry.get("tokens"), list)
		and isinstance(entry.get("bio_tags"), list)
	]
	return valid_entries


def evaluate(
	xml_examples: list[tuple[str, list[XmlEntity]]],
	entries: list[dict],
	show_failures: bool = False,
) -> None:
	total_examples = min(len(xml_examples), len(entries))
	if total_examples == 0:
		print("No comparable examples found.")
		return

	if len(xml_examples) != len(entries):
		print(
			"Warning: example count mismatch. "
			f"YAML examples={len(xml_examples)}, gen_prompt entries={len(entries)}. "
			f"Evaluating first {total_examples}."
		)

	matched_per_category: dict[str, int] = defaultdict(int)
	total_per_category: dict[str, int] = defaultdict(int)

	compared_entities = 0
	matched_entities = 0
	text_mismatch_count = 0
	bio_entity_count = 0
	failures: list[dict[str, str | int]] = []

	for idx in range(total_examples):
		expected_text, xml_entities = xml_examples[idx]
		entry = entries[idx]

		entry_text = entry.get("text", "")
		tokens = entry.get("tokens", [])
		bio_tags = entry.get("bio_tags", [])

		if expected_text != entry_text:
			text_mismatch_count += 1

		if len(tokens) != len(bio_tags):
			continue

		bio_entity_count += sum(1 for tag in bio_tags if isinstance(tag, str) and tag.startswith("B-"))

		for entity in xml_entities:
			total_per_category[entity.category] += 1
			compared_entities += 1

			is_match = entity_matches_bio_by_tokens(
				tokens=tokens,
				bio_tags=bio_tags,
				category=entity.category,
				entity_text=entity.text,
			)

			if is_match:
				matched_per_category[entity.category] += 1
				matched_entities += 1
			elif show_failures:
				failures.append(
					{
						"example_index": idx,
						"entry_id": entry.get("id", "?"),
						"category": entity.category,
						"entity_text": entity.text,
						"context": build_context_snippet(entry_text, entity.text),
					}
				)

	print("XML tag -> BIO validation")
	print("=" * 80)
	print(f"Compared examples: {total_examples}")
	print(f"Examples with text mismatch: {text_mismatch_count}")
	print(f"Compared XML-tag occurrences: {compared_entities}")
	print(f"BIO entity count (B-tags): {bio_entity_count}")
	bio_xml_ratio = (bio_entity_count / compared_entities * 100.0) if compared_entities else 0.0
	print(
		"BIO-vs-XML count ratio: "
		f"{bio_entity_count}/{compared_entities} ({bio_xml_ratio:.2f}%)"
	)
	print()

	print(f"{'Category':<24} {'Matched':>8} {'Total':>8} {'Rate':>10}")
	print("-" * 80)
	for category in sorted(total_per_category):
		total = total_per_category[category]
		matched = matched_per_category.get(category, 0)
		rate = (matched / total * 100.0) if total else 0.0
		print(f"{category:<24} {matched:>8} {total:>8} {rate:>9.2f}%")

	print("-" * 80)
	total_precision = (matched_entities / compared_entities * 100.0) if compared_entities else 0.0
	print(
		f"TOTAL{'':<19} {matched_entities:>8} {compared_entities:>8} {total_precision:>9.2f}%"
	)

	if show_failures:
		print()
		print(f"Failures ({len(failures)}):")
		print("-" * 80)
		for failure in failures:
			print(
				f"idx={failure['example_index']} id={failure['entry_id']} "
				f"category={failure['category']} text={failure['entity_text']}"
			)
			print(f"  context: {failure['context']}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Check whether XML tags in generated YAML response_content correspond "
			"to BIO tags in gen_prompt.json."
		)
	)
	parser.add_argument(
		"--yaml-dir",
		type=Path,
		default=Path("data/raw/generated/dfm-anonymization-gen"),
		help="Directory containing generated YAML files.",
	)
	parser.add_argument(
		"--gen-prompt",
		type=Path,
		default=Path("data/ground_truth/gen_prompt.json"),
		help="Path to gen_prompt.json.",
	)
	parser.add_argument(
		"--hide-failures",
		action="store_true",
		help="Print each unmatched XML-tag occurrence with example/category details.",
	)
	args = parser.parse_args()

	if not args.yaml_dir.exists():
		raise FileNotFoundError(f"YAML directory not found: {args.yaml_dir}")
	if not args.gen_prompt.exists():
		raise FileNotFoundError(f"gen_prompt file not found: {args.gen_prompt}")

	xml_examples = load_yaml_examples(args.yaml_dir)
	entries = load_gen_prompt_entries(args.gen_prompt)
	evaluate(xml_examples, entries, show_failures=not args.hide_failures)


if __name__ == "__main__":
	main()
