import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score


# ---------------------------------------------------------
# CNN1D + CNN2D sonuçlarını oku
# ---------------------------------------------------------

files = sorted(glob.glob("data/*_test_preds.csv"))

results = []

for p in files:
    d = pd.read_csv(p)

    model = (
        "CNN1D"
        if "cnn1d" in os.path.basename(p)
        else "CNN2D"
    )

    for sir, g in d.dropna(subset=["sir_db"]).groupby("sir_db"):
        results.append({
            "model": model,
            "sir": sir,
            "f1": f1_score(
                g.y_true,
                g.y_pred,
                zero_division=0
            ),
            "recall": recall_score(
                g.y_true,
                g.y_pred,
                zero_division=0
            )
        })


# ---------------------------------------------------------
# Energy Threshold sonuçlarını oku
# ---------------------------------------------------------

energy_path = "data/results/energy_baseline_predictions.csv"

energy = pd.read_csv(energy_path)

for sir, g in energy.dropna(subset=["sir_db"]).groupby("sir_db"):
    results.append({
        "model": "Energy",
        "sir": sir,
        "f1": f1_score(
            g.y_true,
            g.y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            g.y_true,
            g.y_pred,
            zero_division=0
        )
    })


# ---------------------------------------------------------
# Özet
# ---------------------------------------------------------

df = pd.DataFrame(results)

summary = (
    df.groupby(["model", "sir"])
      .agg(
          f1_mean=("f1", "mean"),
          f1_std=("f1", "std"),
          recall_mean=("recall", "mean"),
          recall_std=("recall", "std")
      )
      .reset_index()
)

print("\n===== F1 / RECALL vs SIR =====")
print(summary.to_string(index=False))


# ---------------------------------------------------------
# F1 vs SIR grafiği
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

for model in ["Energy", "CNN1D", "CNN2D"]:
    g = summary[summary["model"] == model]

    plt.errorbar(
        g["sir"],
        g["f1_mean"],
        yerr=g["f1_std"],
        marker="o",
        capsize=5,
        label=model
    )

plt.xlabel("SIR (dB)")
plt.ylabel("F1 Score")
plt.title("F1 Score vs SIR")
plt.xticks([0, 5, 10])
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "data/f1_vs_sir.png",
    dpi=300
)

plt.show()

summary.to_csv(
    "data/results/sir_summary.csv",
    index=False
)

print("\nKaydedildi:")
print("data/results/sir_summary.csv")