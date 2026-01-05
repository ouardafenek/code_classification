#!/opt/anaconda3/bin/python
#!/usr/bin/env python3
"""
Baseline multilabel text classification using TF-IDF + Logistic Regression.

Modes:
- none  : no text preprocessing
- nltk  : lemmatization + stemming

CLI-ready, with model & plots saving.
"""

from __future__ import annotations

# =====================
# Standard library
# =====================
import argparse
from pathlib import Path
from typing import List, Tuple

# =====================
# Third-party
# =====================

import joblib
import nltk
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

from utils.utils_train import * 

# =====================
# NLTK setup
# =====================
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# =====================
# Defaults
# =====================
DEFAULT_INPUT_DIR = Path("../data/code_classification_split")
DEFAULT_OUTPUT_DIR = Path("../outputs/baseline (TF-IDF + Logistic Regression OneVsRest)")
DEFAULT_TEXT_COLUMN = "cleaned_description_and_code_source"


# =====================
# NLTK Preprocessor
# =====================

class NLTKPreprocessor:
    """Tokenization + lemmatization + stemming"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def __call__(self, doc: str) -> List[str]:
        tokens = word_tokenize(doc)
        return [
            self.stemmer.stem(self.lemmatizer.lemmatize(t))
            for t in tokens
        ]

# =====================
# Training
# =====================

def build_pipeline(preprocess: str) -> Pipeline:
    if preprocess == "nltk":
        vectorizer = TfidfVectorizer(
            tokenizer=NLTKPreprocessor(),
            token_pattern=None,
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                random_state=42,
            )
        )),
    ])



# =====================
# Main
# =====================

def main(
    input_path: Path,
    output_path: Path,
    text_column: str,
    preprocess: str,
):
    print("\n===== CONFIGURATION =====")
    print(f"Input path   : {input_path}")
    print(f"Output path  : {output_path}")
    print(f"Text column : {text_column}")
    print(f"Preprocess  : {preprocess}")
    print("========================\n")

    X_train_df, y_train, X_test_df, y_test = load_split_data(input_path)

    X_train = X_train_df[text_column]
    X_test = X_test_df[text_column]

    pipeline = build_pipeline(preprocess)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Predicting...")
    y_pred = pipeline.predict(X_test)

    print(f"\nF1 macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    output_path.mkdir(parents=True, exist_ok=True)


    print("Saving model...")
    model_path = output_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


    print("Saving metrics and plots...")
    plot_dir = output_path / "plots"
    plot_dir.mkdir(exist_ok=True)
    save_metrics_and_plots(y_test, y_pred, plot_dir)


# =====================
# CLI
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline multilabel TF-IDF + Logistic Regression (with optional NLTK preprocessing)"
    )

    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text_column", type=str, default=DEFAULT_TEXT_COLUMN)

    parser.add_argument(
        "--preprocess",
        type=str,
        choices=["none", "nltk"],
        default="none",
        help="Text preprocessing mode.",
    )

    

    args = parser.parse_args()

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        text_column=args.text_column,
        preprocess=args.preprocess,
    )
