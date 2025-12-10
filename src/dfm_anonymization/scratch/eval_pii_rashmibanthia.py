import hashlib
import json
import os
import subprocess
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    fbeta_score,
    precision_recall_fscore_support,
)
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

# PII_TYPES = [
#     "NAME_STUDENT",
#     "EMAIL",
#     "USERNAME",
#     "ID_NUM",
#     "PHONE_NUM",
#     "URL_PERSONAL",
#     "STREET_ADDRESS"]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "cpu"
)

competition_name = "pii-detection-removal-from-educational-data"
data_fp = Path(f"data/raw/kagglehub/competitions/{competition_name}").resolve()
data_fp.parent.mkdir(exist_ok=True, parents=True)
if os.system(f"ls {data_fp} > /dev/null") != 0:
    cmd = f"kaggle competitions download -c {competition_name} -p {data_fp.parent}"
    subprocess.run(cmd.split(" "), check=True)
    subprocess.run(f"unzip -q {data_fp}.zip -d {data_fp}".split(" "), check=True)
    # os.system(f"rm {data_fp}.zip")

with open(data_fp / "train.json", "r") as f:
    djson = json.load(f)
djson = djson[:20]
with open(data_fp / "train_reduced.json", "w") as f:
    json.dump(djson, f)

df = pd.read_json(data_fp / "train.json").set_index("document", inplace=False)
classes = {x for xs in list(df["labels"]) for x in xs}
assert classes == {
    "B-EMAIL",
    "B-ID_NUM",
    "B-NAME_STUDENT",
    "B-PHONE_NUM",
    "B-STREET_ADDRESS",
    "B-URL_PERSONAL",
    "B-USERNAME",
    "I-ID_NUM",
    "I-NAME_STUDENT",
    "I-PHONE_NUM",
    "I-STREET_ADDRESS",
    "I-URL_PERSONAL",
    "O",
}

assert df.columns.tolist() == ["text", "tokens", "trailing_whitespace", "labels"]

model_fp = Path(
    kagglehub.model_download(
        "rashmibanthia/pii-data-detection-deberta-v3-large/pyTorch/5folds"
    )
)


class Config:
    model = model_fp
    tokenizer_path = model_fp / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    gradient_checkpointing = False
    max_len = 5000

    batch_size = 4
    num_workers = 2
    num_labels = 13


CFG = Config()

model_paths = [
    f"{model_fp}/microsoft-deberta-v3-large_fold{i}_best.pth" for i in range(5)
]
config_path = f"{model_fp}/config.pth"
config = torch.load(config_path, weights_only=False)
id2label = config.id2label
label2id = config.label2id

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"{model_fp}/tokenizer")


def _file_signature(path: Path) -> str:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return f"{path}:missing"
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def _compute_cache_key(*, model_paths, config_path, dataset_path, subset_size):
    signature_parts = [
        _file_signature(Path(dataset_path)),
        _file_signature(Path(config_path)),
        str(subset_size),
        str(CFG.max_len),
    ]
    signature_parts.extend(_file_signature(Path(p)) for p in model_paths)
    signature = "|".join(signature_parts)
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model, output_hidden_states=True
            )
            self.config.num_labels = 13  # len(all_labels) # <- ??
            self.config.id2label = id2label
            self.config.label2id = label2id
        else:
            self.config = torch.load(config_path, weights_only=False)

        if pretrained:
            self.model = AutoModel.from_pretrained(
                cfg.model, ignore_mismatched_sizes=True, config=self.config
            )
        else:
            self.model = AutoModel.from_config(self.config)

        self.model.resize_token_embeddings(len(CFG.tokenizer))
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.bilstm = nn.LSTM(
            self.config.hidden_size,
            (self.config.hidden_size) // 2,
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

        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

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
        )  # returns ['last_hidden_state', 'hidden_states']
        sequence_output = output[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        gru_output, _ = self.gru(sequence_output)

        lstm_output_new = (
            self.lstm_weights * lstm_output + (1 - self.lstm_weights) * gru_output
        )

        logits = self.fc(lstm_output_new)

        return logits


def chunk_document(tokens, labels):
    max_length = min(CFG.max_len, tokenizer.model_max_length)
    encodings = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_overflowing_tokens=True,
        padding=False,
    )

    token_type_sequences = encodings.get(
        "token_type_ids", [[0] * len(seq) for seq in encodings["input_ids"]]
    )

    chunks = []
    for batch_idx in range(len(encodings["input_ids"])):
        word_ids = encodings.word_ids(batch_index=batch_idx)
        chunk_labels = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                chunk_labels.append(-100)
            elif word_idx == prev_word_idx:
                chunk_labels.append(-100)
            else:
                chunk_labels.append(label2id[labels[word_idx]])
            prev_word_idx = word_idx

        chunk = {
            "input_ids": torch.tensor(
                encodings["input_ids"][batch_idx], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                encodings["attention_mask"][batch_idx], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                token_type_sequences[batch_idx], dtype=torch.long
            ),
            "labels": torch.tensor(chunk_labels, dtype=torch.long),
            "word_ids": word_ids,
        }
        chunks.append(chunk)

    return chunks


true_labels = []
pred_labels = []

subset_df = df

cache_dir = Path("data/cache/evaluations")
cache_dir.mkdir(parents=True, exist_ok=True)
cache_key = _compute_cache_key(
    model_paths=model_paths,
    config_path=config_path,
    dataset_path=data_fp / "train.json",
    subset_size=len(subset_df),
)
cache_file = cache_dir / f"eval_pii_lal_{cache_key}.json"

if cache_file.exists():
    with cache_file.open("r", encoding="utf-8") as f:
        cached = json.load(f)
    true_labels = cached["true_labels"]
    pred_labels = cached["pred_labels"]
    print(f"Loaded cached predictions from {cache_file}")
else:
    model = CustomModel(CFG, config_path=config_path, pretrained=False)
    state = torch.load(
        model_paths[4], map_location=torch.device(device), weights_only=False
    )
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for _, sample in tqdm(
            subset_df.iterrows(),
            total=len(subset_df),
            desc="Evaluating documents",
            unit="doc",
        ):
            tokens = sample["tokens"]
            labels = sample["labels"]
            chunks = chunk_document(tokens, labels)

            for chunk in chunks:
                input_ids = chunk["input_ids"].unsqueeze(0).to(device)
                attention_mask = chunk["attention_mask"].unsqueeze(0).to(device)
                token_type_ids = chunk["token_type_ids"].unsqueeze(0).to(device)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                predictions = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
                labels_tensor = chunk["labels"].tolist()

                for pred_id, label_id, word_idx in zip(
                    predictions, labels_tensor, chunk["word_ids"]
                ):
                    if word_idx is None or label_id == -100:
                        continue
                    true_labels.append(id2label[label_id])
                    pred_labels.append(id2label[pred_id])

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "true_labels": true_labels,
                "pred_labels": pred_labels,
                "metadata": {
                    "cache_key": cache_key,
                    "model_paths": model_paths,
                    "config_path": config_path,
                    "dataset": str(data_fp / "train.json"),
                    "subset_size": len(subset_df),
                },
            },
            f,
        )
    print(f"Saved predictions cache to {cache_file}")


label_order = [id2label[idx] for idx in sorted(id2label.keys())]
precision, recall, f1, support = precision_recall_fscore_support(
    true_labels, pred_labels, labels=label_order, zero_division=0
)


def compute_fbeta(pr, rc, beta):
    beta_sq = beta**2
    denominator = (beta_sq * pr) + rc
    scores = np.zeros_like(pr, dtype=float)
    mask = denominator != 0
    scores[mask] = (1 + beta_sq) * (pr[mask] * rc[mask]) / denominator[mask]
    return scores


per_label_df = pd.DataFrame(
    {
        "label": label_order,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f5": compute_fbeta(precision, recall, beta=5),
        "support": support,
    }
)

weighted_metrics = precision_recall_fscore_support(
    true_labels, pred_labels, average="weighted", zero_division=0
)
macro_metrics = precision_recall_fscore_support(
    true_labels, pred_labels, average="macro", zero_division=0
)
micro_metrics = precision_recall_fscore_support(
    true_labels, pred_labels, average="micro", zero_division=0
)

overall = {
    "accuracy": accuracy_score(true_labels, pred_labels),
    "precision_weighted": weighted_metrics[0],
    "recall_weighted": weighted_metrics[1],
    "f1_weighted": weighted_metrics[2],
    "f5_weighted": fbeta_score(
        true_labels, pred_labels, beta=5, average="weighted", zero_division=0
    ),
    "precision_macro": macro_metrics[0],
    "recall_macro": macro_metrics[1],
    "f1_macro": macro_metrics[2],
    "f5_macro": fbeta_score(
        true_labels, pred_labels, beta=5, average="macro", zero_division=0
    ),
    "f1_micro": micro_metrics[2],
    "f5_micro": fbeta_score(
        true_labels, pred_labels, beta=5, average="micro", zero_division=0
    ),
}

print("Overall Metrics:")
for metric_name, metric_value in overall.items():
    print(f"  {metric_name}: {metric_value:.4f}")

print("\nPer-label Metrics:")
print(
    per_label_df.round(
        {
            "precision": 4,
            "recall": 4,
            "f1": 4,
            "f5": 4,
        }
    ).to_string(index=False)
)

print("\nClassification Report:")
print(
    classification_report(true_labels, pred_labels, labels=label_order, zero_division=0)
)
