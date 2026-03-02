import json
import re
from pathlib import Path
import yaml


def tokenize_text(text):
    """Tokenize text splitting on whitespace and punctuation."""
    # Split on whitespace first
    tokens = []
    for word in text.split():
        # Split punctuation from words
        parts = re.findall(r"\w+|[^\w\s]", word)
        tokens.extend(parts)
    return tokens


def parse_xml_tags(text):
    """Parse XML tags and return clean text with entity positions."""
    entities = []
    clean_text = text
    offset = 0

    # Find all XML tags: <CATEGORY text="string"/>
    pattern = r'<([A-Z_]+)\s+text="([^"]+)"\s*/>'

    for match in re.finditer(pattern, text):
        tag_name = match.group(1)
        entity_text = match.group(2)
        tag_start = match.start() - offset
        tag_end = tag_start + len(entity_text)

        entities.append(
            {"text": entity_text, "tag": tag_name, "start": tag_start, "end": tag_end}
        )

        # Replace tag with just the text
        clean_text = (
            clean_text[: match.start() - offset]
            + entity_text
            + clean_text[match.end() - offset :]
        )
        offset += len(match.group(0)) - len(entity_text)

    return clean_text, entities


def create_bio_tags(tokens, entities, text):
    """Create BIO tags for tokens based on entity positions."""
    bio_tags = ["O"] * len(tokens)

    # Build token position mapping
    token_positions = []
    pos = 0
    for token in tokens:
        # Find token position in text
        token_pos = text.find(token, pos)
        if token_pos != -1:
            token_positions.append((token_pos, token_pos + len(token)))
            pos = token_pos + len(token)
        else:
            token_positions.append((pos, pos))

    # Assign BIO tags based on entities
    for entity in entities:
        start_char = entity["start"]
        end_char = entity["end"]
        tag_name = entity["tag"].lower()

        # Find which tokens overlap with this entity
        tagged_tokens = []
        for i, (tok_start, tok_end) in enumerate(token_positions):
            # Check if token overlaps with entity
            if tok_start < end_char and tok_end > start_char:
                tagged_tokens.append(i)

        # Assign B- and I- tags
        for idx, token_idx in enumerate(tagged_tokens):
            if idx == 0:
                bio_tags[token_idx] = f"B-{tag_name}"
            else:
                bio_tags[token_idx] = f"I-{tag_name}"

    return bio_tags


def process_yaml_files(input_dir, output_file):
    """Process all YAML files and create BIO format JSON."""
    input_path = Path(input_dir)
    yaml_files = sorted(input_path.glob("*.yaml"))

    entries = []
    entry_id = 0

    for yaml_file in yaml_files:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        response_content = data.get("response_content", "")
        if not response_content:
            continue

        # Split by double newline to get separate examples
        examples = [ex.strip() for ex in response_content.split("\n\n") if ex.strip()]

        for example in examples:
            # Parse XML tags
            clean_text, entities = parse_xml_tags(example)

            # Tokenize
            tokens = tokenize_text(clean_text)

            # Create BIO tags
            bio_tags = create_bio_tags(tokens, entities, clean_text)

            # Create entry
            entry = {
                "id": entry_id,
                "text": clean_text,
                "tokens": tokens,
                "bio_tags": bio_tags,
                "metadata": {"language": "danish"},
            }
            entries.append(entry)
            entry_id += 1

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(yaml_files)} YAML files")
    print(f"Created {len(entries)} entries")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    input_dir = "data/raw/generated/dfm-anonymization-gen"
    output_file = "data/ground_truth/generated/generated_prompt.json"
    process_yaml_files(input_dir, output_file)
