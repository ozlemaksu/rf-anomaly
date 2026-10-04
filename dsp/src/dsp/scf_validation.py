from pathlib import Path
import os

_RF_ROOT = Path(os.environ.get("RF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))  # dsp/ klasoru
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, rankdata


CSV_PATH = (_RF_ROOT / "outputs" / "scf_v4" / "scf_features_v4_all_30MHz.csv")


def calculate_auc(x0, x1):
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]

    combined = np.concatenate([x0, x1])
    ranks = rankdata(combined)

    n0 = len(x0)
    n1 = len(x1)

    rank_sum_x1 = np.sum(ranks[n0:])

    u1 = rank_sum_x1 - n1 * (n1 + 1) / 2

    return u1 / (n0 * n1)


def cohens_d(x0, x1):
    n0 = len(x0)
    n1 = len(x1)

    var0 = np.var(x0, ddof=1)
    var1 = np.var(x1, ddof=1)

    pooled_std = np.sqrt(
        ((n0 - 1) * var0 + (n1 - 1) * var1)
        / (n0 + n1 - 2)
    )

    if pooled_std == 0:
        return np.nan

    return (np.mean(x1) - np.mean(x0)) / pooled_std


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\n========================================")
print("DATASET")
print("========================================")

print("Rows:", len(df))
print("OnlyLTE:", (df["label"] == 0).sum())
print("LTE+DSSS:", (df["label"] == 1).sum())

print("\nCOLUMNS:")
print(df.columns.tolist())


# ============================================================
# FEATURES
# ============================================================

features = [
    "target_1p2mhz_comb_score",
    "target_energy_ratio",
    "best_spacing_score",
    "max_to_median",
]


# ============================================================
# OVERALL VALIDATION
# ============================================================

print("\n========================================")
print("OVERALL VALIDATION")
print("========================================")


for feature in features:

    x0 = df[df["label"] == 0][feature].dropna().values
    x1 = df[df["label"] == 1][feature].dropna().values

    auc = calculate_auc(x0, x1)

    u, p = mannwhitneyu(
        x0,
        x1,
        alternative="two-sided"
    )

    d = cohens_d(x0, x1)

    print(f"\nFEATURE: {feature}")

    print("OnlyLTE:")
    print(f"  n      = {len(x0)}")
    print(f"  median = {np.median(x0):.6f}")
    print(f"  mean   = {np.mean(x0):.6f}")

    print("LTE+DSSS:")
    print(f"  n      = {len(x1)}")
    print(f"  median = {np.median(x1):.6f}")
    print(f"  mean   = {np.mean(x1):.6f}")

    print(f"AUC            = {auc:.6f}")
    print(f"Mann-Whitney p = {p:.6e}")
    print(f"Cohen d        = {d:.6f}")


# ============================================================
# FIND SIR COLUMN AUTOMATICALLY
# ============================================================

sir_candidates = [
    "LTE_DSSS_SIR_dB",
    "SIR",
    "sir",
    "SIR_dB",
    "LTE_DSSS_SIR",
]

sir_column = None

for candidate in sir_candidates:
    if candidate in df.columns:
        sir_column = candidate
        break


# ============================================================
# SIR-STRATIFIED VALIDATION
# ============================================================

print("\n========================================")
print("SIR-STRATIFIED VALIDATION")
print("========================================")

if sir_column is None:

    print("\nSIR column was not found in the SCF CSV.")
    print("SIR-stratified analysis skipped.")
    print("The overall validation above is still valid.")

else:

    print(f"\nUsing SIR column: {sir_column}")

    df[sir_column] = pd.to_numeric(
        df[sir_column],
        errors="coerce"
    )

    for sir in [0, 5, 10]:

        print(f"\n---------- SIR = {sir} dB ----------")

        combined = df[
            (df["label"] == 1) &
            (df[sir_column] == sir)
        ]

        normal = df[df["label"] == 0]

        print("Combined recordings:", len(combined))

        for feature in features:

            x0 = normal[feature].dropna().values
            x1 = combined[feature].dropna().values

            if len(x1) == 0:
                print(f"{feature}: no data")
                continue

            auc = calculate_auc(x0, x1)

            print(
                f"{feature}: "
                f"median={np.median(x1):.6f}, "
                f"AUC={auc:.6f}"
            )


# ============================================================
# TOP OUTLIERS
# ============================================================

print("\n========================================")
print("TOP OUTLIERS")
print("========================================")


for feature in [
    "target_1p2mhz_comb_score",
    "target_energy_ratio",
    "best_spacing_score",
]:

    print(f"\n--- {feature} ---")

    columns = ["file", "label"]

    if "LTE_SNR_dB" in df.columns:
        columns.append("LTE_SNR_dB")

    if sir_column is not None:
        columns.append(sir_column)

    columns.append(feature)

    print(
        df[columns]
        .sort_values(feature, ascending=False)
        .head(10)
        .to_string(index=False)
    )


print("\n========================================")
print("VALIDATION COMPLETE")
print("========================================")