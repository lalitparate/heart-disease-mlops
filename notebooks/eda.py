"""
Exploratory Data Analysis — Heart Disease UCI Dataset
Run: PYTHONPATH=. python notebooks/eda.py
Outputs saved to: models/plots/eda/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "heart.csv")
OUT_DIR   = os.path.join("models", "plots", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"0": "#2ecc71", "1": "#e74c3c", "main": "#3498db"}

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex (0=F, 1=M)",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting BP (mmHg)",
    "chol":     "Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise Angina",
    "oldpeak":  "ST Depression",
    "slope":    "ST Slope",
    "ca":       "Major Vessels",
    "thal":     "Thalassemia",
    "target":   "Heart Disease",
}

NUMERIC_FEATURES     = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Heart Disease EDA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df.replace("?", np.nan, inplace=True)
for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["target"] = (df["target"] > 0).astype(int) if df["target"].max() > 1 else df["target"]

print(f"\nDataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
print(f"\nData types:\n{df.dtypes.to_string()}")
print(f"\nBasic stats:\n{df.describe().round(2).to_string()}")


# ── 1. Missing Value Analysis ─────────────────────────────────────────────────
print("\n[1/5] Missing value analysis...")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Missing Value Analysis", fontsize=15, fontweight="bold")

# Heatmap
sns.heatmap(df.isnull(), yticklabels=False, cbar=True,
            cmap="YlOrRd", ax=axes[0])
axes[0].set_title("Missing Value Heatmap")
axes[0].set_xlabel("Features")

# Bar chart
if not missing_df.empty:
    missing_df["Missing %"].plot(kind="bar", color="#e74c3c",
                                  edgecolor="white", ax=axes[1])
    axes[1].set_title("Missing Values per Feature (%)")
    axes[1].set_ylabel("Missing %")
    axes[1].tick_params(axis="x", rotation=45)
else:
    axes[1].text(0.5, 0.5, "No Missing Values Found!",
                 ha="center", va="center", fontsize=14,
                 color="#2ecc71", fontweight="bold",
                 transform=axes[1].transAxes)
    axes[1].set_title("Missing Values per Feature")

plt.tight_layout()
path = os.path.join(OUT_DIR, "01_missing_values.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {path}")
if missing_df.empty:
    print("   ✓ No missing values found in dataset")
else:
    print(f"   Missing values:\n{missing_df.to_string()}")


# ── 2. Class Distribution ─────────────────────────────────────────────────────
print("\n[2/5] Class distribution...")

counts = df["target"].value_counts().sort_index()
labels = ["No Disease (0)", "Heart Disease (1)"]
colors = [COLORS["0"], COLORS["1"]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Target Class Distribution", fontsize=15, fontweight="bold")

# Bar chart
bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
axes[0].set_title("Class Counts")
axes[0].set_ylabel("Number of Patients")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha="center", va="bottom", fontweight="bold")

# Pie chart
axes[1].pie(counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class Proportion")

plt.tight_layout()
path = os.path.join(OUT_DIR, "02_class_distribution.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {path}")
print(f"   Class 0 (No Disease) : {counts[0]} ({counts[0]/len(df)*100:.1f}%)")
print(f"   Class 1 (Disease)    : {counts[1]} ({counts[1]/len(df)*100:.1f}%)")


# ── 3. Histograms — Numeric Features ─────────────────────────────────────────
print("\n[3/5] Histograms for numeric features...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Histograms — Numeric Features by Class", fontsize=15, fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(NUMERIC_FEATURES):
    ax = axes[i]
    for cls, color, label in [(0, COLORS["0"], "No Disease"), (1, COLORS["1"], "Disease")]:
        data = df[df["target"] == cls][col].dropna()
        ax.hist(data, bins=20, alpha=0.6, color=color, label=label, edgecolor="white")
    ax.set_title(FEATURE_LABELS[col])
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    # Add mean lines
    for cls, color in [(0, COLORS["0"]), (1, COLORS["1"])]:
        mean_val = df[df["target"] == cls][col].mean()
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.5)

# Hide unused subplot
axes[-1].set_visible(False)

plt.tight_layout()
path = os.path.join(OUT_DIR, "03_histograms_numeric.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {path}")


# ── 4. Categorical Feature Distributions ─────────────────────────────────────
print("\n[4/5- part A] Categorical feature distributions...")

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("Categorical Features vs Heart Disease", fontsize=15, fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(CATEGORICAL_FEATURES):
    ax = axes[i]
    ct = df.groupby([col, "target"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=ax, color=[COLORS["0"], COLORS["1"]],
            edgecolor="white", width=0.7)
    ax.set_title(FEATURE_LABELS[col])
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["No Disease", "Disease"], fontsize=8)

plt.tight_layout()
path = os.path.join(OUT_DIR, "04_categorical_distributions.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {path}")


# ── 5. Correlation Heatmap ────────────────────────────────────────────────────
print("\n[4/5 - part B] Correlation heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Correlation Analysis", fontsize=15, fontweight="bold")

# Full correlation heatmap
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=axes[0], annot_kws={"size": 8})
axes[0].set_title("Full Feature Correlation Matrix")
axes[0].tick_params(axis="x", rotation=45)

# Correlation with target only
target_corr = corr["target"].drop("target").sort_values()
colors_bar = [COLORS["1"] if v > 0 else COLORS["0"] for v in target_corr]
axes[1].barh(target_corr.index, target_corr.values, color=colors_bar, edgecolor="white")
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_title("Feature Correlation with Target")
axes[1].set_xlabel("Pearson Correlation")
for i, (val, name) in enumerate(zip(target_corr.values, target_corr.index)):
    axes[1].text(val + (0.01 if val >= 0 else -0.01), i,
                 f"{val:.2f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=9)

plt.tight_layout()
path = os.path.join(OUT_DIR, "05_correlation_heatmap.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {path}")


# ── 6. Feature Relationship Analysis ─────────────────────────────────────────
print("\n[5/5] Feature relationship analysis...")

# Box plots — numeric features by class
fig, axes = plt.subplots(1, 5, figsize=(20, 6))
fig.suptitle("Feature Distribution by Class (Box Plots)", fontsize=15, fontweight="bold")

for i, col in enumerate(NUMERIC_FEATURES):
    data0 = df[df["target"] == 0][col].dropna()
    data1 = df[df["target"] == 1][col].dropna()
    bp = axes[i].boxplot([data0, data1],
                          patch_artist=True,
                          labels=["No Disease", "Disease"],
                          medianprops={"color": "black", "linewidth": 2})
    bp["boxes"][0].set_facecolor(COLORS["0"])
    bp["boxes"][1].set_facecolor(COLORS["1"])
    for patch in bp["boxes"]:
        patch.set_alpha(0.7)
    axes[i].set_title(FEATURE_LABELS[col])
    axes[i].tick_params(axis="x", rotation=15)

plt.tight_layout()
path = os.path.join(OUT_DIR, "06_boxplots_by_class.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {path}")

# Scatter plot — Age vs Max Heart Rate coloured by class
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feature Relationship Analysis", fontsize=15, fontweight="bold")

for cls, color, label in [(0, COLORS["0"], "No Disease"), (1, COLORS["1"], "Disease")]:
    subset = df[df["target"] == cls]
    axes[0].scatter(subset["age"], subset["thalach"],
                    c=color, label=label, alpha=0.6, edgecolors="white", s=60)
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Max Heart Rate")
axes[0].set_title("Age vs Max Heart Rate")
axes[0].legend()

# Scatter — Cholesterol vs Resting BP
for cls, color, label in [(0, COLORS["0"], "No Disease"), (1, COLORS["1"], "Disease")]:
    subset = df[df["target"] == cls]
    axes[1].scatter(subset["chol"], subset["trestbps"],
                    c=color, label=label, alpha=0.6, edgecolors="white", s=60)
axes[1].set_xlabel("Cholesterol (mg/dl)")
axes[1].set_ylabel("Resting BP (mmHg)")
axes[1].set_title("Cholesterol vs Resting BP")
axes[1].legend()

plt.tight_layout()
path = os.path.join(OUT_DIR, "07_scatter_relationships.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {path}")

# Pair plot for numeric features
fig, axes = plt.subplots(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES),
                          figsize=(18, 16))
fig.suptitle("Pairplot — Numeric Features", fontsize=15, fontweight="bold", y=1.01)

for i, col_y in enumerate(NUMERIC_FEATURES):
    for j, col_x in enumerate(NUMERIC_FEATURES):
        ax = axes[i][j]
        if i == j:
            for cls, color in [(0, COLORS["0"]), (1, COLORS["1"])]:
                data = df[df["target"] == cls][col_x].dropna()
                ax.hist(data, bins=15, alpha=0.6, color=color, edgecolor="white", density=True)
        else:
            for cls, color in [(0, COLORS["0"]), (1, COLORS["1"])]:
                subset = df[df["target"] == cls]
                ax.scatter(subset[col_x], subset[col_y],
                           c=color, alpha=0.4, s=15, edgecolors="none")
        if i == len(NUMERIC_FEATURES) - 1:
            ax.set_xlabel(col_x, fontsize=8)
        if j == 0:
            ax.set_ylabel(col_y, fontsize=8)
        ax.tick_params(labelsize=6)

plt.tight_layout()
path = os.path.join(OUT_DIR, "08_pairplot.png")
fig.savefig(path, dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {path}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EDA Complete — All plots saved to:", OUT_DIR)
print("=" * 60)
print("\nPlots generated:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  ✓ {f}")

print("\nKey Findings:")
print(f"  • Dataset: {len(df)} patients, {df.shape[1]} features")
print(f"  • Disease prevalence: {df['target'].mean()*100:.1f}%")
print(f"  • Mean age: {df['age'].mean():.1f} years")
print(f"  • Top correlated features with target:")
target_corr_abs = df.corr(numeric_only=True)["target"].drop("target").abs().sort_values(ascending=False)
for feat, val in target_corr_abs.head(5).items():
    print(f"      {feat:<12} r = {val:.3f}")
