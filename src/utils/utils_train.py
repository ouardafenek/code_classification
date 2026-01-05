from pathlib import Path
from typing import Tuple

# Data Loading 
import pandas as pd
import numpy as np

# Metrics & Plots
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



# =====================
# Data loading
# =====================
def load_split_data(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading datasets...")
    X_train = pd.read_parquet(input_dir / "X_train.parquet")
    y_train = pd.read_parquet(input_dir / "y_train.parquet")
    X_test = pd.read_parquet(input_dir / "X_test.parquet")
    y_test = pd.read_parquet(input_dir / "y_test.parquet")
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test




# =====================
# Metrics & Plots
# =====================
def _generate_metrics_reports_and_plots(metrics_df, y_true_names, y_true, y_pred, output_path: Path):
    """Fonction interne commune pour sauvegarder les rapports et générer les graphiques."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Sauvegarde du rapport texte
    with open(output_path / "classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=y_true_names, zero_division=0))

    # 2. Plot Precision / Recall / F1
    metrics_df = metrics_df.sort_values("f1", ascending=False)
    x = np.arange(len(metrics_df))
    width = 0.22 

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, metrics_df["precision"], width, label="Precision", color='#4e79a7')
    rects2 = ax.bar(x,         metrics_df["recall"],    width, label="Recall",    color='#f28e2b')
    rects3 = ax.bar(x + width, metrics_df["f1"],        width, label="F1",        color='#59a14f')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_ylabel('Scores')
    ax.set_title("Precision / Recall / F1 per class", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["label"], rotation=45, ha="right")
    ax.set_ylim(0, 1.1) 
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path / "metrics_per_label.png", dpi=300)
    plt.close()

    # 3. Plot support per class
    plt.figure(figsize=(10, 5))
    bars = plt.bar(metrics_df["label"], metrics_df["support"], color='#76b7b2')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of samples")
    plt.title("Support per class")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "support_per_label.png", dpi=300)
    plt.close()

    print(f"Metrics and plots saved in {output_path}")


def save_metrics_and_plots(y_true: pd.DataFrame, y_pred, output_path: Path):
    """Version pour DataFrame standard."""
    report = classification_report(y_true, y_pred, target_names=y_true.columns.tolist(), zero_division=0, output_dict=True)
    
    metrics_df = pd.DataFrame({
        "label": y_true.columns,
        "precision": [report[label]["precision"] for label in y_true.columns],
        "recall": [report[label]["recall"] for label in y_true.columns],
        "f1": [report[label]["f1-score"] for label in y_true.columns],
        "support": [report[label]["support"] for label in y_true.columns],
    })
    
    _generate_metrics_reports_and_plots(metrics_df, y_true.columns.tolist(), y_true, y_pred, output_path)


def save_metrics_and_plots_bert(y_true, y_pred, label_columns, output_path: Path):
    """Version pour CodeBERT / Multi-label tensors."""
    # Calcul du support manuellement pour les tensors si nécessaire, ou via classification_report
    report = classification_report(y_true, y_pred, target_names=label_columns, zero_division=0, output_dict=True)
    
    metrics_df = pd.DataFrame({
        "label": label_columns,
        "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=None, zero_division=0),
        "support": [report[label]["support"] for label in label_columns],
    })
    
    _generate_metrics_reports_and_plots(metrics_df, label_columns, y_true, y_pred, output_path)