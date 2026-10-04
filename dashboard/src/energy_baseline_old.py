from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


BASE = Path(".")

INDEX_FILE = BASE / "data" / "dataset_index_split.csv"
FROZEN_FILE = BASE / "data" / "frozen_indices.json"
DATASET_DIR = BASE / "data" / "MATLAB_Dataset" / "IQ"


def read_iq(path):
    with open(path, "rb") as f:
        b = f.read()

    header = np.frombuffer(b, dtype="<i4", count=2)

    if header[0] != 2:
        raise ValueError(f"Beklenmeyen header: {header.tolist()}")

    n = int(header[1])

    data = np.frombuffer(
        b,
        dtype="<f4",
        count=2 * n,
        offset=8
    )

    if data.size != 2 * n:
        raise ValueError("Dosya header bilgisindeki örnek sayısından kısa.")

    i = data[0::2]
    q = data[1::2]

    return i, q


def signal_power(i, q):
    return float(np.mean(i.astype(np.float64) ** 2 + q.astype(np.float64) ** 2))


def main():

    df = pd.read_csv(INDEX_FILE)

    with open(FROZEN_FILE, "r") as f:
        frozen = json.load(f)

    test_groups = set(frozen["test"])
    val_groups = set(frozen["val"])

    val_df = df[df["group"].isin(val_groups)].copy()
    test_df = df[df["group"].isin(test_groups)].copy()

    print("=== ENERGY BASELINE ===")
    print("Validation kayıtları:", len(val_df))
    print("Test kayıtları:", len(test_df))

    print("\nValidation enerji hesaplanıyor...")

    val_scores = []
    val_labels = []

    for _, row in val_df.iterrows():

        path = DATASET_DIR / str(row["file"]).replace(".csv", "")

        i, q = read_iq(path)

        score = signal_power(i, q)

        val_scores.append(score)
        val_labels.append(int(row["label"]))

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)

    thresholds = np.unique(
        np.quantile(
            val_scores,
            np.linspace(0.01, 0.99, 200)
        )
    )

    best_threshold = thresholds[0]
    best_f1 = -1

    for threshold in thresholds:

        pred = (val_scores > threshold).astype(int)

        score = f1_score(
            val_labels,
            pred,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print("Validation F1:", best_f1)
    print("Seçilen threshold:", best_threshold)

    print("\nTest enerji hesaplanıyor...")

    test_scores = []
    test_labels = []

    for _, row in test_df.iterrows():

        path = DATASET_DIR / str(row["file"]).replace(".csv", "")

        i, q = read_iq(path)

        score = signal_power(i, q)

        test_scores.append(score)
        test_labels.append(int(row["label"]))

    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    test_pred = (test_scores > best_threshold).astype(int)

    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred, zero_division=0)
    recall = recall_score(test_labels, test_pred, zero_division=0)
    f1 = f1_score(test_labels, test_pred, zero_division=0)
    auc = roc_auc_score(test_labels, test_scores)

    cm = confusion_matrix(
        test_labels,
        test_pred,
        labels=[0, 1]
    )

    print("\n=== TEST SONUÇLARI ===")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC-AUC:", auc)

    print("\nConfusion Matrix:")
    print(cm)

    result = {
        "model": "Energy Baseline",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "threshold": float(best_threshold),
        "confusion_matrix": cm.tolist()
    }

    output_dir = BASE / "data" / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "energy_baseline_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    prediction_df = test_df[
        ["file", "label", "sir_db", "group"]
    ].copy()

    prediction_df["energy"] = test_scores
    prediction_df["y_pred"] = test_pred

    prediction_df.to_csv(
        output_dir / "energy_baseline_predictions.csv",
        index=False
    )

    print("\nSonuçlar kaydedildi:")
    print(output_dir / "energy_baseline_metrics.json")
    print(output_dir / "energy_baseline_predictions.csv")


if __name__ == "__main__":
    main()