import json
import glob
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------
# CNN1D / CNN2D sonuçları
# ---------------------------------------------------------

rows = []

files = sorted(glob.glob("data/*_test_preds.csv"))

for p in files:
    df = pd.read_csv(p)

    name = os.path.basename(p)

    if "cnn1d" in name:
        model = "CNN1D"
    elif "cnn2d" in name:
        model = "CNN2D"
    else:
        continue

    y_true = df["y_true"]
    y_pred = df["y_pred"]
    y_prob = df["y_prob"]

    rows.append({
        "model": model,
        "seed": name.split("_s")[1].split("_")[0],
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, y_prob
        ),
    })


# ---------------------------------------------------------
# Energy Threshold
# ---------------------------------------------------------

with open(
    "data/results/energy_baseline_metrics.json",
    "r",
    encoding="utf-8"
) as f:
    energy = json.load(f)


rows.append({
    "model": "Energy",
    "seed": "-",
    "accuracy": energy["test_accuracy"],
    "precision": energy["test_precision"],
    "recall": energy["test_recall"],
    "f1": energy["test_f1"],
    "roc_auc": energy["test_roc_auc"],
})


# ---------------------------------------------------------
# Tablo
# ---------------------------------------------------------

result = pd.DataFrame(rows)

result = result[
    [
        "model",
        "seed",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
]

print("\n===== MODEL COMPARISON =====")
print(result.to_string(index=False))

result.to_csv(
    "data/results/model_comparison.csv",
    index=False
)

print(
    "\nKaydedildi:"
    "\ndata/results/model_comparison.csv"
)