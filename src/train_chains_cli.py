#!/opt/anaconda3/bin/python
#!/usr/bin/env python3
"""
Multilabel text classification using Ensemble of Classifier Chains
with LightGBM + TF-IDF.

CLI-ready, with model & plots saving.
"""

from __future__ import annotations

# =====================
# Standard library
# =====================
import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
import joblib
import warnings

# =====================
# Third-party
# =====================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# =====================
# Custom Classes and Functions 
# =====================
from utils.chain_ensemble_pipeline import ChainEnsemblePipeline
from utils.utils_train import load_split_data, save_metrics_and_plots

# =====================
# Defaults
# =====================
DEFAULT_INPUT_DIR = Path("../data/code_classification_split")
DEFAULT_OUTPUT_DIR = Path(
    "../outputs/chains (TF-IDF + Ensemble of Classifier Chains with LightGBM)"
)
DEFAULT_TEXT_COLUMN = "cleaned_description_and_code_source"
DEFAULT_N_CHAINS = 50
DEFAULT_MAX_FEATURES = 10000
DEFAULT_THRESHOLD = 0.5


# =====================
# Training ensemble chains
# =====================
def train_ensemble_chains(
    X_train: pd.Series,
    y_train: pd.DataFrame,
    X_test: pd.Series,
    y_test: pd.DataFrame,
    n_chains: int,
    max_features: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer, List[ClassifierChain]]:

    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 4),
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    probabilities_sum = np.zeros((X_test_vec.shape[0], y_test.shape[1]))
    chains_list: List[ClassifierChain] = []

    print(f"Training ensemble of {n_chains} Classifier Chains...")

    for i in range(n_chains):
        lgbm = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42 + i,
            n_jobs=-1,
            verbose=-1,
        )

        chain = ClassifierChain(lgbm, order="random", random_state=i)
        chain.fit(X_train_vec, y_train)

        chains_list.append(chain)

        probs = chain.predict_proba(X_test_vec)
        probabilities_sum += probs

        print(f" -> Chain {i + 1}/{n_chains} done.")

    avg_probabilities = probabilities_sum / n_chains
    y_pred = (avg_probabilities >= threshold).astype(int)

    return avg_probabilities, y_pred, tfidf, chains_list

# =====================
# Main
# =====================
def main(
    input_path: Path,
    output_path: Path,
    text_column: str,
    n_chains: int,
    max_features: int,
    threshold: float,
):
    print("\n===== CONFIGURATION =====")
    print(f"Input path     : {input_path}")
    print(f"Output path    : {output_path}")
    print(f"Text column    : {text_column}")
    print(f"Num chains     : {n_chains}")
    print(f"Max features   : {max_features}")
    print(f"Threshold      : {threshold}")
    print("========================\n")

    X_train_df, y_train, X_test_df, y_test = load_split_data(input_path)
    X_train = X_train_df[text_column]
    X_test = X_test_df[text_column]

    avg_probs, y_pred, tfidf, chains_list = train_ensemble_chains(
        X_train,
        y_train,
        X_test,
        y_test,
        n_chains=n_chains,
        max_features=max_features,
        threshold=threshold,
    )

    print(f"\nF1 macro: {f1_score(y_test, y_pred, average='macro'):.4f}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    print("Saving pipeline model...")
    pipeline = ChainEnsemblePipeline(
        tfidf=tfidf,
        chains=chains_list,
        threshold=threshold,
    )

    model_path = output_path / "chain_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Pipeline model saved to: {model_path}")

    # Save metrics & plots
    print("Saving metrics and plots...")
    plot_dir = output_path / "plots"
    plot_dir.mkdir(exist_ok=True)
    save_metrics_and_plots(y_test, y_pred, plot_dir)

# =====================
# CLI
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multilabel ensemble of classifier chains with LightGBM + TF-IDF"
    )

    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text_column", type=str, default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--n_chains", type=int, default=DEFAULT_N_CHAINS)
    parser.add_argument("--max_features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    args = parser.parse_args()

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        text_column=args.text_column,
        n_chains=args.n_chains,
        max_features=args.max_features,
        threshold=args.threshold,
    )
