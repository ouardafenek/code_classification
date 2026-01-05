#!/opt/anaconda3/bin/python
#!/usr/bin/env python3

"""
Multilabel training script for CodeBERT.

- Configurable via CLI (argparse)
- Training / evaluation pipeline
- Multilabel classification with class imbalance handling
"""

from __future__ import annotations

# =====================
# Standard library
# =====================
import argparse
from pathlib import Path
from typing import Tuple, List

# =====================
# Third-party
# =====================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

# =====================
# Costum Classes and Functions 
# =====================
from utils.code_bert_for_multi_label import CodeBERTForMultiLabel
from utils.utils_train import * 
from settings.globals import * 



from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================
# Default configuration
# =====================

MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512

DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 30
DEFAULT_LR = 2e-5
DEFAULT_THRESHOLD = 0.5

DEFAULT_INPUT_DIR = Path("../data/code_classification_split")
DEFAULT_OUTPUT_DIR = Path("../outputs/multilabel_CodeBERT")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N_LABELS = len(LABEL_COLUMNS)

TEXT_COLUMN = "cleaned_description_and_code_source"


# =====================
# Dataset
# =====================

class CodeDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: torch.Tensor,
        tokenizer: AutoTokenizer,
        max_len: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }



# =====================
# Training utilities
# =====================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        loss = criterion(logits, batch["labels"].to(DEVICE))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
            )
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    return np.vstack(all_preds), np.vstack(all_labels)

# =====================
# Main
# =====================

def main(
    input_path: Path = DEFAULT_INPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    output_path: Path = DEFAULT_OUTPUT_DIR,
    threshold: float = DEFAULT_THRESHOLD,
):
    print("\n===== TRAINING CONFIG =====")
    print(f"Input path   : {input_path}")
    print(f"Batch size   : {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs       : {epochs}")
    print(f"Threshold    : {threshold}")
    print(f"Output path  : {output_path}")
    print("===========================\n")

    # -------- Data --------
    X_train_df, y_train_df, X_test_df, y_test_df = load_split_data(input_path)

    train_texts = X_train_df[TEXT_COLUMN].tolist()
    test_texts = X_test_df[TEXT_COLUMN].tolist()

    y_train = torch.tensor(y_train_df[LABEL_COLUMNS].values, dtype=torch.float)
    y_test = torch.tensor(y_test_df[LABEL_COLUMNS].values, dtype=torch.float)

    # -------- Class imbalance --------
    support = y_train.sum(dim=0)
    n_samples = y_train.shape[0]
    pos_weight = ((n_samples - support) / support).clamp(min=1.0)
    pos_weight = pos_weight.to(torch.float).to(DEVICE)

    print("Positive weights per label:", pos_weight)

    # -------- Datasets --------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = CodeDataset(train_texts, y_train, tokenizer, MAX_LEN)
    test_dataset = CodeDataset(test_texts, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # -------- Model --------
    model = CodeBERTForMultiLabel(MODEL_NAME, N_LABELS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # -------- Training loop --------
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        preds, labels = evaluate(model, test_loader, threshold)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"F1 micro: {f1_score(labels, preds, average='micro'):.4f} | "
            f"F1 macro: {f1_score(labels, preds, average='macro'):.4f}"
        )

    # -------- Save --------
    output_path.mkdir(parents=True, exist_ok=True)
    

    print("Saving Model...")
    output_model_path = output_path / "model"
    torch.save(model.state_dict(), output_model_path / "model.pt")
    tokenizer.save_pretrained(output_model_path)
    print(f"Model saved to {output_model_path}")

    # -------- Final evaluation --------
    y_pred, y_true = evaluate(model, test_loader, threshold)

    print("\n=== GLOBAL METRICS ===")
    print(f"F1 micro   : {f1_score(y_true, y_pred, average='micro'):.4f}")
    print(f"F1 macro   : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 weighted: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Precision  : {precision_score(y_true, y_pred, average='micro'):.4f}")
    print(f"Recall     : {recall_score(y_true, y_pred, average='micro'):.4f}")
    print(f"Hamming    : {hamming_loss(y_true, y_pred):.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(labels, preds, target_names=LABEL_COLUMNS, zero_division=0))
    print("Saving metrics and plots...")
    plot_dir = output_path / "plots"
    plot_dir.mkdir(exist_ok=True)
    save_metrics_and_plots_bert(y_true, y_pred, LABEL_COLUMNS, plot_dir)


# =====================
# CLI
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CodeBERT for multilabel classification on code + text data."
    )

    parser.add_argument(
        "--input_path",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Path to the input directory containing the train/test splits "
            "(X_train.parquet, y_train.parquet, X_test.parquet, y_test.parquet). "
            f"Default: {DEFAULT_INPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Batch size used for training and evaluation. "
            "Larger values use more GPU memory. "
            f"Default: {DEFAULT_BATCH_SIZE}"
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=(
            "Learning rate for the AdamW optimizer. "
            "Typical values for CodeBERT are in the range [1e-5, 5e-5]. "
            f"Default: {DEFAULT_LR}"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            "Number of training epochs. "
            "Increasing this may improve performance but can lead to overfitting. "
            f"Default: {DEFAULT_EPOCHS}"
        ),
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where the fine-tuned model, tokenizer, and evaluation plots "
            "will be saved. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Decision threshold applied to sigmoid outputs during evaluation "
            "to convert probabilities into binary predictions. "
            "Must be in the range [0, 1]. "
            f"Default: {DEFAULT_THRESHOLD}"
        ),
    )
        
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        batch_size=args.bs,
        learning_rate=args.lr,
        epochs=args.epochs,
        output_path=args.output_path,
        threshold=args.threshold,
    )
