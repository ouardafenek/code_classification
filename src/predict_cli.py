#!/opt/anaconda3/bin/python
from __future__ import annotations

# =====================
# Standard library
# =====================
import argparse
from pathlib import Path
import warnings

# =====================
# Third-party
# =====================
import numpy as np
import joblib
from transformers import AutoTokenizer
import torch 

# =====================
# Costum classes and functions
# =====================
from utils.chain_ensemble_pipeline import ChainEnsemblePipeline
from utils.code_bert_for_multi_label import CodeBERTForMultiLabel
from utils.utils_predict import * 
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
    """
    model_path = joblib file containing:
    - TfidfVectorizer
    - ClassifierChain 
    """
    model = joblib.load(model_path)

    X = [input_text]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        probs = model.predict(X)[0]

    preds = probs >= 0.5
    return probs, preds




# =====================
# CLI
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict tags using CodeBERT or Classifier Chain"
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
        "--input",
        type=Path,
        required=True,
        help="Path to JSON input file",
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


    preprocessed_text, ground_tags = load_input_text(args.input)

    if args.model_type == "codebert":
        probs, preds = predict_codebert(
            model_dir=args.model_path,
            input_text=preprocessed_text,
            threshold=args.threshold,
            max_len=512
        )
    else:
        probs, preds = predict_chain(
            model_path=args.model_path,
            input_text=preprocessed_text,
        )

    print("\n=== PREDICTION RESULTS ===")
    for label, p, pred in zip(LABEL_COLUMNS, probs, preds):
        status = "✔" if pred else "✘"
        print(f"{status} {label:15s} | prob = {float(p):.4f}")

    filtered_ground_tags = [tag for tag in ground_tags if tag in LABEL_COLUMNS]
    print(f"\nThe ground truth tags in the sample from the 8 below are => {filtered_ground_tags}")

    print(f"\nUsed Model: {args.model_type}")
