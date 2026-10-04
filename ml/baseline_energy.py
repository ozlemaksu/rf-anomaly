"""Klasik baseline: enerji (guc) esigi. Yapay zekanin ne kattigini gostermek icin ilk satir.

Mantik: her pencerenin ortalama gucu hesaplanir; esik yalnizca val setinden secilir
(F1'i en yuksek yapan deger); test setinde bir kez uygulanir.
Anomali (DSSS) ek guc getirdigi icin yon sabittir: "guc buyukse anomali".
Val AUC'si 0.5'ten kucukse yon DEGISTIRILMEZ, yalnizca uyari yazilir
(80-100 ornekli val setine bakarak yon secmek sizinti gibi calisir).

Ornek:
    python ml/baseline_energy.py --x data/X_iq.npy --y data/y.npy --splits data/splits.npz
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def power(X):
    """(N, 2, L) -> (N,) ortalama guc."""
    return (X.astype(np.float64) ** 2).sum(axis=1).mean(axis=1)


def best_threshold(score, y):
    """Val setinde F1'i en buyuk yapan esik (aday: 200 nicelik)."""
    cands = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 200)))
    best_t, best_f1 = cands[0], -1.0
    for t in cands:
        f1 = f1_score(y, score > t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t), float(best_f1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out_dir", default="ml/results")
    p.add_argument("--tag", default="baseline_energy")
    a = p.parse_args()

    X = np.load(a.x, mmap_mode="r")  # diske eslenir, tamami belleğe alinmaz
    y = np.load(a.y).astype(np.int64)
    sp = np.load(a.splits)
    va, te = sp["val_idx"], sp["test_idx"]
    if X.ndim != 3 or X.shape[1] != 2:
        raise SystemExit(f"X_iq sekli (N, 2, L) olmali, gelen {X.shape}")

    s_va = power(X[va])
    auc_va = roc_auc_score(y[va], s_va)
    if auc_va < 0.5:
        print(f"UYARI: val AUC {auc_va:.3f} < 0.5. DSSS guc eklemeli, yon beklenenin tersi.")
        print("       Veri pencere basina normalize edilmis olabilir; guc esigi bu durumda anlamsizdir.")
    thr, f1_va = best_threshold(s_va, y[va])
    print(f"esik (val'dan): {thr:.4g}   val F1: {f1_va:.3f}   val AUC: {auc_va:.3f}")

    s_te = power(X[te])
    pred = s_te > thr
    y_te = y[te]
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_te, s_te) if len(np.unique(y_te)) == 2 else float("nan")

    # tek pencere cikarim suresi (ms)
    one = X[te[:1]]
    t0 = time.perf_counter()
    for _ in range(200):
        _ = power(one) > thr
    lat = (time.perf_counter() - t0) / 200 * 1000

    metrics = {
        "model": a.tag,
        "acc": float(accuracy_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "auc": float(auc),
        "latency_ms": float(lat),
        "n_params": 0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(thr),
        "n_test": int(len(te)),
    }
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"idx": te, "y_true": y_te, "p_anomaly": s_te}).to_csv(
        os.path.join(a.out_dir, f"{a.tag}_test_preds.csv"), index=False
    )
    print(f"TEST  acc {metrics['acc']:.3f}  F1 {metrics['f1']:.3f}  AUC {metrics['auc']:.3f}  "
          f"({metrics['latency_ms']:.3f} ms/pencere)  [TN {tn} FP {fp} FN {fn} TP {tp}]")


if __name__ == "__main__":
    main()
