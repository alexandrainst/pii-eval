# Load model directly
import json
from pathlib import Path
from typing import List, Tuple
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "saattrupdan/nbailab-base-ner-scandi"
dataset_name = "gen_address"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "cpu"
)

def bio_to_spans(
        pred_ents: List[str],
        tokens: torch.Tensor,
        offsets: List[Tuple[int, int]],
        text: str
    ):

    token_ids = tokens[0].tolist()[1:-1]

    spans = []
    active_label = None
    active_start = None
    active_end = None

    for (start, end), tag in zip(offsets, pred_ents):
        if tag == 'O':
            if active_label:
                spans.append({
                    'label': active_label,
                    'start': active_start,
                    'end': active_end,
                    'text': text[active_start:active_end]
                })
                active_label = None
            continue

        prefix, label = tag.split('-', 1) if '-' in tag else ('B', tag)
        if prefix == 'B' and active_label == label:
            prefix = 'I'
        if prefix == 'I' and active_label == label:
            active_end = end
            continue

        if active_label:
            spans.append({
                'label': active_label,
                'start': active_start,
                'end': active_end,
                'text': text[active_start:active_end]
            })

        active_label = label
        active_start = start
        active_end = end

    if active_label:
        spans.append({
            'label': active_label,
            'start': active_start,
            'end': active_end,
            'text': text[active_start:active_end]
        })

    return spans


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
model.to(device)
dataset_path = Path(f"data/span/{dataset_name}.json")
dataset = json.loads(dataset_path.read_text())
id2labels = model.config.id2label

tokenized_sents = []
out_data = []
for document in tqdm(dataset):
    document_id = document["document"]
    text = document['text']
    tokenized = tokenizer(
        text, return_tensors="pt",
        return_offsets_mapping=True)
    input_ids = tokenized["input_ids"].to(device)
    preds = model(input_ids)[0]
    preds = torch.argmax(preds, dim=2)
    preds = preds[0].tolist()[1:-1]
    pred_ents = [id2labels[p] for p in preds]
    offset_mapping = tokenized["offset_mapping"][0].tolist()[1:-1]
    spans = bio_to_spans(
        pred_ents, input_ids, offset_mapping, text)
    out_data.append(
        {"document": document_id, "text": text, "spans": spans})

out_path = Path(f"data/infer/{dataset_name}/scandiner.json")
out_path.parent.mkdir(exist_ok=True, parents=True)
out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=4))

# predictions = []
# for (input, mask) in tokenized_sents:
#     preds = model(input)[0]
#     preds = torch.argmax(preds, dim=2)
#     preds = preds[0].tolist()[1:-1]
#     assert(len(preds) == len(mask))
#     pred_ents = [id2labels[p] for p, m in zip(preds, mask) if m]
#     assert(len(pred_ents) == sum(mask))
#     predictions.append(pred_ents)

pass
