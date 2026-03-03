import json
from pathlib import Path


def visualize_bio_tags(entry):
    """Convert BIO tags and tokens to a human-readable format."""
    tokens = entry["tokens"]
    bio_tags = entry["bio_tags"]

    print(f"\n{'=' * 80}")
    print(f"Entry ID: {entry['id']}")
    print(f"Text: {entry['text']}")
    print(f"\n{'Token':<20} {'BIO Tag':<20}")
    print(f"{'-' * 40}")

    for token, tag in zip(tokens, bio_tags):
        print(f"{token:<20} {tag:<20}")

    # Also show entities grouped
    print(f"\n{'Entities found:'}")
    current_entity = []
    current_tag = None

    for token, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            if current_entity:
                entity_type = current_tag.replace("B-", "").replace("I-", "")
                print(f"  [{entity_type}]: {' '.join(current_entity)}")
            current_entity = [token]
            current_tag = tag
        elif tag.startswith("I-") and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                entity_type = current_tag.replace("B-", "").replace("I-", "")
                print(f"  [{entity_type}]: {' '.join(current_entity)}")
                current_entity = []
                current_tag = None

    # Don't forget the last entity
    if current_entity:
        entity_type = current_tag.replace("B-", "").replace("I-", "")
        print(f"  [{entity_type}]: {' '.join(current_entity)}")


def main():
    """Load and visualize the first few entries from the generated data."""
    data_file = Path("data/ground_truth/gen_prompt.json")

    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])

    print(f"Total entries: {len(entries)}")
    print("\nShowing first 5 entries:")

    # Display first 5 entries
    for entry in entries[:5]:
        visualize_bio_tags(entry)

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
