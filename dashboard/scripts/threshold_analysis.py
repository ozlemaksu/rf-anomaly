import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = DATA / "results"

files = [
    "cnn1d_s1_test_preds.csv",
    "cnn1d_s2_test_preds.csv",
    "cnn1d_s3_test_preds.csv",
    "cnn2d_s1_test_preds.csv",
    "cnn2d_s2_test_preds.csv",
    "cnn2d_s3_test_preds.csv",
]

thresholds = np.arange(0.10, 0.91, 0.05)

all_results = []

for filename in files:

    path = DATA / filename

    df = pd.read_csv(path)

    y_true = df["y_true"].values
    y_prob = df["y_prob"].values

    model = "CNN1D" if "cnn1d" in filename else "CNN2D"
    seed = int(filename.split("_s")[1].split("_")[0])

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        all_results.append({
            "model": model,
            "seed": seed,
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            )
        })

result = pd.DataFrame(all_results)

output = RESULTS / "threshold_analysis.csv"
result.to_csv(output, index=False)

print("Threshold analizi tamamlandı.")
print(f"Kaydedildi: {output}")

print("\nEn iyi F1 değerleri:")

best = (
    result
    .sort_values("f1", ascending=False)
    .groupby(["model", "seed"])
    .head(1)
)

print(
    best[
        [
            "model",
            "seed",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
)