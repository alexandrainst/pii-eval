import json
from pathlib import Path
from typing import List

import kagglehub
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# PII_TYPES = [
#     "NAME_STUDENT",
#     "EMAIL",
#     "USERNAME",
#     "ID_NUM",
#     "PHONE_NUM",
#     "URL_PERSONAL",
#     "STREET_ADDRESS"]

dataset = "dansk_ner"

data_fp = Path(f"data/span/da/{dataset}.json").resolve()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "cpu"
)

with open(data_fp, "r") as f:
    ds: List[dict] = json.load(f)

assert set(ds[0].keys()) == {"document", "full_text", "spans"}
model_fp = Path(
    kagglehub.model_download(
        "rashmibanthia/pii-data-detection-deberta-v3-large/pyTorch/5folds"
    )
)
tokenizer = AutoTokenizer.from_pretrained(f"{model_fp}/tokenizer")
model_fold_fps = [
    model_fp / f"/microsoft-deberta-v3-large_fold{i}_best.pth" for i in range(5)
]
config = torch.load(model_fp / "config.pth", weights_only=False)
id2label = config.id2label
label2id = config.label2id

id2label = {int(k): v for k, v in id2label.items()}
label2id = {k: int(v) for k, v in label2id.items()}
config.id2label = id2label
config.label2id = label2id
model_fold_fps = [
    model_fp / f"microsoft-deberta-v3-large_fold{i}_best.pth" for i in range(5)
]


class CFG:
    model = model_fp
    tokenizer = tokenizer
    gradient_checkpointing = False
    max_len = min(
        5000,
        tokenizer.model_max_length
        if tokenizer.model_max_length and tokenizer.model_max_length < 10**6
        else 5000,
    )
    num_labels = len(id2label)


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config.__class__.from_dict(config.to_dict())
        self.model = AutoModel.from_config(self.config)
        self.model.resize_token_embeddings(len(CFG.tokenizer))
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            dropout=self.config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True,
        )
        self.gru = nn.GRU(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            dropout=self.config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_weights = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.fc = nn.Linear(self.config.hidden_size, CFG.num_labels)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(output[0])
        lstm_output, _ = self.bilstm(sequence_output)
        gru_output, _ = self.gru(sequence_output)
        sequence_output = (
            self.lstm_weights * lstm_output + (1 - self.lstm_weights) * gru_output
        )
        return self.fc(sequence_output)


def _tokenize_text(text: str):
    max_length = CFG.max_len
    encodings = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        padding=False,
    )
    token_type_sequences = encodings.get(
        "token_type_ids",
        [[0] * len(seq) for seq in encodings["input_ids"]],
    )
    chunks = []
    for idx in range(len(encodings["input_ids"])):
        chunks.append(
            {
                "input_ids": torch.tensor(
                    encodings["input_ids"][idx], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    encodings["attention_mask"][idx], dtype=torch.long
                ),
                "token_type_ids": torch.tensor(
                    token_type_sequences[idx], dtype=torch.long
                ),
                "offset_mapping": encodings["offset_mapping"][idx],
            }
        )
    return chunks


def _decode_predictions(token_labels_offsets):
    spans = []
    current = None
    for label, (start, end) in token_labels_offsets:
        if start is None or end is None or start == end:
            continue
        if label == "O":
            if current:
                spans.append(current)
                current = None
            continue
        if "-" in label:
            prefix, entity = label.split("-", 1)
        else:
            prefix, entity = "B", label
        if prefix == "B" or not current or current["label"] != entity:
            if current:
                spans.append(current)
            current = {"label": entity, "start": int(start), "end": int(end)}
        else:
            current["end"] = int(end)
    if current:
        spans.append(current)
    if not spans:
        return spans
    merged = [spans[0]]
    for span in spans[1:]:
        last = merged[-1]
        if last["label"] == span["label"] and last["end"] >= span["start"]:
            last["end"] = max(last["end"], span["end"])
        else:
            merged.append(span)
    return merged


models = []
for fold_path in model_fold_fps:
    state = torch.load(fold_path, map_location=device, weights_only=False)
    model = CustomModel()
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    models.append(model)


def infer_text(text: str):
    token_predictions = []
    chunks = _tokenize_text(text)
    with torch.no_grad():
        for chunk in chunks:
            input_ids = chunk["input_ids"].unsqueeze(0).to(device)
            attention_mask = chunk["attention_mask"].unsqueeze(0).to(device)
            token_type_ids = chunk["token_type_ids"].unsqueeze(0).to(device)
            logits_sum = None
            for model in models:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits_sum = logits if logits_sum is None else logits_sum + logits
            predictions = logits_sum.argmax(dim=-1).squeeze(0).cpu().tolist()
            for pred_id, offsets in zip(predictions, chunk["offset_mapping"]):
                if offsets is None:
                    continue
                start, end = offsets
                if start == end:
                    continue
                token_predictions.append(
                    (id2label[int(pred_id)], (int(start), int(end)))
                )
    spans = _decode_predictions(token_predictions)
    if not spans:
        return spans
    text_len = len(text)
    for span in spans:
        start_idx = span["start"]
        end_idx = span["end"]
        if start_idx < 0:
            start_idx = 0
        elif start_idx > text_len:
            start_idx = text_len
        if end_idx < start_idx:
            end_idx = start_idx
        elif end_idx > text_len:
            end_idx = text_len
        span["start"] = start_idx
        span["end"] = end_idx
        span["text"] = text[start_idx:end_idx]
    return spans


results = []
for sample in tqdm(ds, desc="Inferring", unit="doc"):
    spans = infer_text(sample["full_text"])
    results.append(
        {
            "document": sample["document"],
            "full_text": sample["full_text"],
            "spans": spans,
        }
    )

output_path = Path(f"data/infer/{dataset}/rashmibanthia.json").resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=True)

reduced_path = (output_path.parent / "reduced" / f"{output_path.stem}.json").resolve()
reduced_path.parent.mkdir(parents=True, exist_ok=True)
with open(reduced_path, "w") as f:
    json.dump(results[:5], f, indent=True)
