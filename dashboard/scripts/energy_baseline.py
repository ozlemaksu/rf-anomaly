from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRE = DATA / "preprocessed"
RESULTS = DATA / "results"

RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. Verileri yükle
# ---------------------------------------------------------

X = np.load(PRE / "X_iq.npy")
y = np.load(PRE / "y.npy")
sir = np.load(PRE / "sir_db.npy")

frozen = np.load(PRE / "frozen_window_indices.npz")

train_idx = frozen["train_idx"]
val_idx = frozen["val_idx"]
test_idx = frozen["test_idx"]


# ---------------------------------------------------------
# 2. Her window için enerji hesapla
# ---------------------------------------------------------

# X shape: (N, 2, samples)
# Kanal 0 = I
# Kanal 1 = Q
#
# Energy = mean(I^2 + Q^2)

energy = np.mean(
    X[:, 0, :] ** 2 + X[:, 1, :] ** 2,
    axis=1
)


# ---------------------------------------------------------
# 3. Validation setinde en iyi threshold'u bul
# ---------------------------------------------------------

val_energy = energy[val_idx]
val_y = y[val_idx]

best_threshold = None
best_f1 = -1

# Validation'daki farklı enerji değerlerini threshold adayı olarak kullan
thresholds = np.unique(val_energy)

for threshold in thresholds:
    val_pred = (val_energy >= threshold).astype(int)

    score = f1_score(
        val_y,
        val_pred,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold


# ---------------------------------------------------------
# 4. Test setinde tahmin
# ---------------------------------------------------------

test_energy = energy[test_idx]
test_y = y[test_idx]
test_sir = sir[test_idx]

test_pred = (
    test_energy >= best_threshold
).astype(int)


# ---------------------------------------------------------
# 5. Test metrikleri
# ---------------------------------------------------------

accuracy = accuracy_score(test_y, test_pred)
precision = precision_score(test_y, test_pred, zero_division=0)
recall = recall_score(test_y, test_pred, zero_division=0)
f1 = f1_score(test_y, test_pred, zero_division=0)

try:
    auc = roc_auc_score(test_y, test_energy)
except ValueError:
    auc = None

cm = confusion_matrix(test_y, test_pred)


# ---------------------------------------------------------
# 6. Sonuçları yazdır
# ---------------------------------------------------------

print("\n===== ENERGY THRESHOLD BASELINE =====")

print(f"Train windows : {len(train_idx)}")
print(f"Val windows   : {len(val_idx)}")
print(f"Test windows  : {len(test_idx)}")

print(f"\nBest threshold : {best_threshold:.8f}")
print(f"Validation F1  : {best_f1:.6f}")

print("\n--- TEST ---")
print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1        : {f1:.6f}")

if auc is not None:
    print(f"ROC-AUC   : {auc:.6f}")

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------------------------
# 7. Prediction CSV
# ---------------------------------------------------------

pred_df = pd.DataFrame({
    "idx": test_idx,
    "y_true": test_y,
    "y_pred": test_pred,
    "energy": test_energy,
    "sir_db": test_sir,
})

pred_path = RESULTS / "energy_baseline_predictions.csv"

pred_df.to_csv(
    pred_path,
    index=False
)


# ---------------------------------------------------------
# 8. Metrics JSON
# ---------------------------------------------------------

metrics = {
    "model": "Energy Threshold",
    "threshold": float(best_threshold),
    "validation_f1": float(best_f1),
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "test_roc_auc": None if auc is None else float(auc),
    "confusion_matrix": cm.tolist(),
    "train_windows": int(len(train_idx)),
    "val_windows": int(len(val_idx)),
    "test_windows": int(len(test_idx)),
}

metrics_path = RESULTS / "energy_baseline_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)


print("\nDosyalar oluşturuldu:")
print(pred_path)
print(metrics_path)