"""
Model training script with MLflow experiment tracking.
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Performs hyperparameter tuning and logs everything to MLflow.
"""

from src.preprocess import (
    load_data,
    split_features_target,
    fit_and_save_preprocessor,
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")


warnings.filterwarnings("ignore")

DATA_PATH   = os.path.join("data", "heart.csv")
MODELS_DIR  = "models"
BEST_PATH   = os.path.join(MODELS_DIR, "best_model.joblib")
PLOTS_DIR   = os.path.join("models", "plots")
EXPERIMENT  = "heart_disease_classification"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Evaluation helper ────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
    }, y_pred, y_proba


def save_confusion_matrix(y_test, y_pred, name):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_roc_curve(y_test, y_proba, name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_{name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ── Model definitions + param grids ─────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        },
    }


# ── Main training loop ───────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  Heart Disease MLOps — Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Target distribution:\n{y.value_counts().to_string()}\n")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Fit and save preprocessor
    preprocessor = fit_and_save_preprocessor(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # 4. MLflow setup
    mlflow.set_experiment(EXPERIMENT)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_auc   = 0.0
    best_model = None
    best_name  = ""
    models     = get_models()

    for model_name, cfg in models.items():
        print(f"\n── Training: {model_name} ──")
        with mlflow.start_run(run_name=model_name):

            # GridSearchCV tuning
            gs = GridSearchCV(
                cfg["model"], cfg["params"],
                cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
            )
            gs.fit(X_train_t, y_train)
            tuned = gs.best_estimator_

            # CV score
            cv_scores = cross_val_score(
                tuned, X_train_t, y_train, cv=cv, scoring="roc_auc"
            )

            # Test evaluation
            metrics, y_pred, y_proba = evaluate(tuned, X_test_t, y_test)

            # Log params & metrics
            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("cv_roc_auc_mean", round(cv_scores.mean(), 4))
            mlflow.log_metric("cv_roc_auc_std",  round(cv_scores.std(),  4))

            # Log plots
            cm_path  = save_confusion_matrix(y_test, y_pred, model_name)
            roc_path = save_roc_curve(y_test, y_proba, model_name)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)

            # Log model
            mlflow.sklearn.log_model(tuned, artifact_path="model")

            print(f"  Best params : {gs.best_params_}")
            print(f"  Test metrics: {metrics}")
            print(f"  CV AUC      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Track best
            if metrics["roc_auc"] > best_auc:
                best_auc   = metrics["roc_auc"]
                best_model = tuned
                best_name  = model_name

    # 5. Save best model
    joblib.dump(best_model, BEST_PATH)
    print(f"\n{'=' * 60}")
    print(f"  Best model : {best_name}  (ROC-AUC = {best_auc:.4f})")
    print(f"  Saved      → {BEST_PATH}")
    print(f"{'=' * 60}")
    return best_model


if __name__ == "__main__":
    train()
