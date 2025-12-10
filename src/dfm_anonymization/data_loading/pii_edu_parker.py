# from
# https://www.kaggle.com/code/deeparker/pii-detection-with-deberta

# Import packages
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import Counter
import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gc
import re
import random
from itertools import chain

# Set random seeds for reproducibility


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_value = 42
set_seed(seed_value)

# Capture the start time
start_time = time.time()
print("Start Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

base_path = "data/raw/kaggle/input/pii-detection-removal-from-educational-data"

# Load dataset
dataset = datasets.load_dataset("json", data_files=f"{base_path}/train.json")

# Convert to DataFrame and preprocess
df = dataset["train"].to_pandas()
all_labels = [label for sublist in df["labels"] for label in sublist]
unique_labels = list(set(all_labels))
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {label: i for i, label in enumerate(unique_labels)}
all_labels = list(label2id.keys())  # Get all unique labels from mapping
num_labels = len(all_labels)  # Update the num_labels

# Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "cpu"
)
model.to(device)

# Split Dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = datasets.Dataset.from_pandas(train_df)
test_dataset = datasets.Dataset.from_pandas(test_df)
dataset_dict = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})

print("Dataset length:", len(dataset))
print(dataset)

df = pd.read_json(f"{base_path}/train.json")
print(df.head())

print()


def format_text(text):
    # Add paragraph breaks
    formatted_text = text.replace("\n\n", "\n\n<p>\n\n")

    # Add bullet points to list items
    formatted_text = re.sub(r"•\s", "\n- ", formatted_text)

    # Handle remaining single newlines
    formatted_text = re.sub(r"(?<!\n)\n(?!\n)", " ", formatted_text)

    # Remove leading and trailing spaces
    formatted_text = re.sub(r"\s+\n", "\n", formatted_text)
    formatted_text = re.sub(r"\n\s+", "\n", formatted_text)

    return formatted_text


# Example usage
raw_text = df["full_text"][0]

print(format_text(raw_text))

# Count the frequency of each label in the 'train' dataset
label_freq = Counter(chain(*train_dataset["labels"]))

# Display the frequency of each label
for label, freq in label_freq.items():
    print(f"Label: {label}, Frequency: {freq}")

# Define function to tokenize and align labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_offsets_mapping=True,
    )

    batch_original_tokens = []
    batch_tokenized_tokens = []
    batch_label_ids = []
    batch_input_ids = []
    batch_attention_masks = []
    batch_token_type_ids = []

    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        original_tokens = examples["tokens"][i]
        tokenized_tokens = tokenizer.convert_ids_to_tokens(
            tokenized_inputs["input_ids"][i]
        )

        previous_word_idx = None
        label_ids = []
        original_token_list = []
        tokenized_token_list = []
        input_id_list = []
        attention_mask_list = []
        token_type_id_list = []

        for j, word_idx in enumerate(word_ids):
            if word_idx is None:
                label_ids.append(-100)
                current_original_token = ""  # Special token
            elif word_idx == previous_word_idx:
                label_ids.append(-100)
                current_original_token = ""  # Subword token
            else:
                label_ids.append(label2id[label[word_idx]])
                current_original_token = original_tokens[word_idx]

            original_token_list.append(current_original_token)
            tokenized_token_list.append(tokenized_tokens[j])
            input_id_list.append(tokenized_inputs["input_ids"][i][j])
            attention_mask_list.append(tokenized_inputs["attention_mask"][i][j])
            if "token_type_ids" in tokenized_inputs:
                token_type_id_list.append(tokenized_inputs["token_type_ids"][i][j])
            else:
                token_type_id_list.append(0)
            previous_word_idx = word_idx  # Update for the next iteration

        batch_original_tokens.append(original_token_list)
        batch_tokenized_tokens.append(tokenized_token_list)
        batch_label_ids.append(label_ids)
        batch_input_ids.append(input_id_list)
        batch_attention_masks.append(attention_mask_list)
        batch_token_type_ids.append(token_type_id_list)

    # Include BERT-required columns
    return {
        "original_tokens": batch_original_tokens,
        "tokenized_tokens": batch_tokenized_tokens,
        "labels": batch_label_ids,
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_masks,
        "token_type_ids": batch_token_type_ids,
    }


# Tokenize
tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

gc.collect()

# List of columns to keep
columns_to_keep = ["labels", "input_ids", "attention_mask", "token_type_ids"]

# Function to remove unnecessary columns


def remove_unnecessary_columns(dataset, columns_to_keep):
    return dataset.remove_columns(
        [column for column in dataset.column_names if column not in columns_to_keep]
    )


# Apply the function to both train and test datasets
tokenized_datasets["train"] = remove_unnecessary_columns(
    tokenized_datasets["train"], columns_to_keep
)
tokenized_datasets["test"] = remove_unnecessary_columns(
    tokenized_datasets["test"], columns_to_keep
)

# Calculate class weights
class_weights = compute_class_weight(
    "balanced", classes=np.unique(all_labels), y=all_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Define Focal Loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        F_loss = ((1 - pt) ** self.gamma) * BCE_loss
        return F_loss.mean()


# Define data collator
data_collator = DataCollatorForTokenClassification(tokenizer)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[label] for label in doc if label != -100] for doc in labels
    ]
    true_predictions = [
        [id2label[pred] for pred, label in zip(doc, labels[i]) if label != -100]
        for i, doc in enumerate(predictions)
    ]

    # Flatten the lists
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    true_predictions_flat = [item for sublist in true_predictions for item in sublist]

    # Calculate metrics
    results = precision_recall_fscore_support(
        true_labels_flat, true_predictions_flat, average="weighted"
    )

    # Compute classification report
    class_report = classification_report(
        # Explicitly set labels and handle zero division
        true_labels_flat,
        true_predictions_flat,
        labels=all_labels,
        zero_division=0,
    )

    print("Classification Report:\n", class_report)
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2],
        "accuracy": (results[2] * results[1]),
    }


# Custom Trainer with custom loss function


class CustomTrainer(Trainer):
    def __init__(self, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, *_args, **_kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# Choose the loss function (WeightedLoss or FocalLoss)
loss_fn = FocalLoss(gamma=2, alpha=class_weights)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none",
)

# Trainer
trainer = CustomTrainer(
    loss_fn=loss_fn,
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
