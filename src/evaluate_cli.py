#!/opt/anaconda3/bin/python
from __future__ import annotations

# =====================
# Standard library
# =====================
import argparse
from pathlib import Path
from typing import List
import json

# =====================
# Third-party
# =====================
import numpy as np
import torch
import torch.nn as nn
import joblib
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# =====================
# Costum classes and functions
# =====================
from utils.chain_ensemble_pipeline import ChainEnsemblePipeline
from utils.code_bert_for_multi_label import CodeBERTForMultiLabel
from utils.utils_predict import *  # load_input_text peut être réutilisé
from settings.globals import *

# =====================
# Configuration
# =====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_TYPE = "codebert"
DEFAULT_MODEL_PATH_CODEBERT = Path("../outputs/multilabel_CodeBERT/model")
DEFAULT_MODEL_PATH_CHAINS = Path("../outputs/chains (TF-IDF + Ensemble of Classifier Chains with LightGBM)/chain_pipeline.joblib")


# =====================
# CodeBERT model
# =====================
def predict_codebert(
    model_dir: Path,
    input_text: str,
    threshold: float,
    max_len: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = CodeBERTForMultiLabel(
        model_name="microsoft/codebert-base",
        n_labels=len(LABEL_COLUMNS),
    )
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    encoding = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            encoding["input_ids"].to(DEVICE),
            encoding["attention_mask"].to(DEVICE),
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    preds = probs >= threshold
    return probs, preds


# =====================
# Classifier Chain
# =====================
def predict_chain(
    model_path: Path,
    input_text: str,
):
    model = joblib.load(model_path)
    X = [input_text]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        probs = model.predict(X)[0]

    preds = probs >= 0.5
    return probs, preds


# =====================
# Evaluation
# =====================
def load_ground_truth(json_file: Path):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    tags = data.get("tags", [])
    y_true = [1 if label in tags else 0 for label in LABEL_COLUMNS]
    return y_true


def evaluate_folder(model_type: str, model_path: Path, input_folder: Path, threshold: float):
    y_trues = []
    y_preds = []

    json_files = list(input_folder.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to evaluate.")

    for file in json_files:
        preprocessed_text, tags = load_input_text(file)  # retourne texte et tags
        y_true = [1 if label in tags else 0 for label in LABEL_COLUMNS]

        if model_type == "codebert":
            probs, preds = predict_codebert(
                model_dir=model_path,
                input_text=preprocessed_text,
                threshold=threshold,
                max_len=512
            )
        else:
            probs, preds = predict_chain(
                model_path=model_path,
                input_text=preprocessed_text,
            )

        y_trues.append(y_true)
        y_preds.append(preds.astype(int))

    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)

    # ======= Métriques globales =======
    metrics_global = {
        "precision_micro": precision_score(y_trues, y_preds, average="micro", zero_division=0),
        "recall_micro": recall_score(y_trues, y_preds, average="micro", zero_division=0),
        "f1_micro": f1_score(y_trues, y_preds, average="micro", zero_division=0),
        "precision_macro": precision_score(y_trues, y_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_trues, y_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_trues, y_preds, average="macro", zero_division=0),
    }

    # ======= Rapport par label =======
    report_dict = classification_report(
        y_trues,
        y_preds,
        target_names=LABEL_COLUMNS,
        output_dict=True,
        zero_division=0, 
    )

    return metrics_global, report_dict


# =====================
# CLI
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model on folder of JSON files"
    )

    parser.add_argument(
        "--model_type",
        choices=["codebert", "chain"],
        default=DEFAULT_MODEL_TYPE,
        help="Which model to use for prediction",
    )

    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Path to model directory (CodeBERT) or .joblib file (Chain). "
             "If not provided, a default path is used depending on model_type.",
    )

    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Path to folder containing JSON files to evaluate",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for multilabel decision (CodeBERT only)",
    )

    args = parser.parse_args()

    if args.model_path is None:
        if args.model_type == "chain":
            args.model_path = DEFAULT_MODEL_PATH_CHAINS
        else:
            args.model_path = DEFAULT_MODEL_PATH_CODEBERT

    metrics_global, report_dict = evaluate_folder(
        model_type=args.model_type,
        model_path=args.model_path,
        input_folder=args.input_folder,
        threshold=args.threshold,
    )

    print("\n============ EVALUATION RESULTS ============")
    for k, v in metrics_global.items():
        print(f"{k:15s}: {v:.4f}")

    # Afficher par label
    print("\nScores per Label:")
    for label in LABEL_COLUMNS:
        r = report_dict[label]
        print(f"{label:15s} | P={r['precision']:.2f}   |  "
              f"R={r['recall']:.2f}   |  F1={r['f1-score']:.2f}   |  Support={r['support']}")
    print("\n\n")
