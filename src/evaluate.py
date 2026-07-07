"""
Evaluation utilities — metrics, plots, and reporting helpers.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)

PLOTS_DIR = os.path.join("models", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
    }


def print_classification_report(y_true, y_pred):
    print(classification_report(
        y_true, y_pred,
        target_names=["No Disease", "Disease"]
    ))


def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"], ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    if save:
        fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_proba, model_name: str, save: bool = True) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1f77b4")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right"); plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_{model_name.lower().replace(' ', '_')}.png")
    if save:
        fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names: list, model_name: str, save: bool = True) -> str:
    """Works for Random Forest and XGBoost."""
    if not hasattr(model, "feature_importances_"):
        return ""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_features[::-1], sorted_importances[::-1],
                   color="#2ecc71", edgecolor="white")
    ax.set_xlabel("Feature Importance"); ax.set_title(f"Feature Importance — {model_name}")
    for bar, val in zip(bars, sorted_importances[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"fi_{model_name.lower().replace(' ', '_')}.png")
    if save:
        fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
