"""Egitim betigi (B rolu).

Ornekler (repo kokunden):
    python src/train.py --model cnn1d --x data/X_iq.npy   --y data/y.npy --splits data/splits.npz
    python src/train.py --model cnn2d --x data/X_spec.npy --y data/y.npy --splits data/splits.npz

Asiri ogrenme testi (egitim kaybi sifira yaklasmali, yoksa mimaride/veride sorun var):
    python src/train.py --model cnn1d --x ... --y ... --splits ... --subset 32 --patience 100

Ciktilar (--out_dir, varsayilan results/):
    <tag>_best.pt            en iyi val kaybindaki agirliklar
    <tag>_metrics.json       test metrikleri (acc, F1, AUC, gecikme, parametre sayisi, karisiklik matrisi)
    <tag>_test_preds.csv     idx, y_true, p_anomaly  (C kisisi grafikleri bundan cizer)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from models import CNN1D, CNN2D


def channel_stats(X, idx, axes, chunk=64):
    """Kanal basina ortalama ve std; yalnizca verilen (egitim) indekslerinden, parca parca (float64)."""
    s1 = s2 = 0.0
    n = 0
    for k in range(0, len(idx), chunk):
        b = np.asarray(X[idx[k:k + chunk]], dtype=np.float64)
        s1 = s1 + b.sum(axis=axes, keepdims=True)
        s2 = s2 + (b * b).sum(axis=axes, keepdims=True)
        n += b.size // s1.size
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0.0)) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)


def make_loader(X, y, bs, shuffle, drop_last=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)


@torch.no_grad()
def predict(model, loader, loss_fn, dev):
    """Dongu sonunda (ortalama kayip, P(anomali) dizisi, gercek etiket dizisi)."""
    model.eval()
    total, probs, ys = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        out = model(xb)
        total += loss_fn(out, yb).item() * len(yb)
        probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        ys.append(yb.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(probs), np.concatenate(ys)


@torch.no_grad()
def latency_ms(model, sample, n=50):
    """Tek ornek icin CPU cikarim suresi (ms, ortalama)."""
    model = model.to("cpu").eval()
    x = torch.from_numpy(np.ascontiguousarray(sample[:1]))
    for _ in range(5):
        model(x)
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    return (time.perf_counter() - t0) / n * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn1d", "cnn2d"], required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset", type=int, default=0, help="yalnizca ilk N egitim ornegi (asiri ogrenme testi)")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--tag", default="", help="cikti dosya oneki (varsayilan: model adi)")
    a = p.parse_args()

    tag = a.tag or a.model
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.load(a.x, mmap_mode="r")  # diske eslenir; yalnizca bolmeler belleğe alinir
    y = np.load(a.y).astype(np.int64)
    sp = np.load(a.splits)
    tr, va, te = sp["train_idx"], sp["val_idx"], sp["test_idx"]

    # --- erken, anlasilir hatalar ---
    want_ndim = 3 if a.model == "cnn1d" else 4
    if X.ndim != want_ndim:
        raise SystemExit(
            f"{a.model} icin X {want_ndim} boyutlu olmali "
            f"({'(N, 2, L)' if want_ndim == 3 else '(N, 1, F, T)'}), gelen sekil {X.shape}. "
            "Dogru dosyayi (--x) verdiginizden emin olun."
        )
    if len(X) != len(y):
        raise SystemExit(f"X ({len(X)}) ve y ({len(y)}) ornek sayisi farkli")
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        if len(idx) == 0 or idx.min() < 0 or idx.max() >= len(X):
            raise SystemExit(f"{name}_idx bos veya aralik disi. Once: python src/check_data.py")

    if a.subset:
        # Sinif dengeli rastgele alt kume (indeksler dosya sirasina gore dizili, ilk N hep tek sinif olabilir)
        rs = np.random.RandomState(a.seed)
        per = max(1, a.subset // 2)
        pick = []
        for c in (0, 1):
            pool = tr[y[tr] == c]
            pick.append(rs.choice(pool, size=min(per, len(pool)), replace=False))
        tr = np.sort(np.concatenate(pick))
        print(f"SUBSET modu: {len(tr)} egitim ornegi (sinif basina en cok {per})")

    # Normalizasyon istatistigi yalnizca egitim setinden (sizintiyi onler)
    axes = (0, 2) if a.model == "cnn1d" else (0, 2, 3)
    mu, sd = channel_stats(X, tr, axes)

    def split_array(idx):
        """Bolmeyi belleğe alir ve yerinde normalize eder (ek kopya yok)."""
        A = np.asarray(X[idx], dtype=np.float32)
        A -= mu
        A /= sd
        return A

    Xtr, Xva, Xte = split_array(tr), split_array(va), split_array(te)
    xshape = (len(X),) + tuple(Xtr.shape[1:])

    counts = np.bincount(y[tr], minlength=2)
    if counts.min() == 0:
        raise SystemExit(f"egitim setinde tek sinif var (LTE {counts[0]}, LTE+DSSS {counts[1]})")
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model = (CNN1D() if a.model == "cnn1d" else CNN2D(in_ch=Xtr.shape[1])).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"{tag}: {n_params:,} parametre, cihaz {dev}, X {xshape}, seed {a.seed}")

    tr_ld = make_loader(Xtr, y[tr], a.bs, True, drop_last=len(tr) > a.bs)
    va_ld = make_loader(Xva, y[va], a.bs, False)
    te_ld = make_loader(Xte, y[te], a.bs, False)

    best, best_ep, wait = float("inf"), 0, 0
    ckpt = os.path.join(a.out_dir, f"{tag}_best.pt")
    for ep in range(a.epochs):
        model.train()
        run, seen = 0.0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(yb)
            seen += len(yb)
        v_loss, v_p, v_y = predict(model, va_ld, loss_fn, dev)
        v_f1 = f1_score(v_y, v_p > 0.5, zero_division=0)
        print(f"ep {ep:02d}  train_loss {run / seen:.4f}  val_loss {v_loss:.4f}  val_F1 {v_f1:.3f}")
        if v_loss < best:
            best, best_ep, wait = v_loss, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= a.patience:
                print("erken durdurma")
                break

    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    _, t_p, t_y = predict(model, te_ld, loss_fn, dev)
    pred = t_p > 0.5
    tn, fp, fn, tp = confusion_matrix(t_y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(t_y, t_p) if len(np.unique(t_y)) == 2 else float("nan")
    metrics = {
        "model": tag,
        "seed": a.seed,
        "acc": float(accuracy_score(t_y, pred)),
        "f1": float(f1_score(t_y, pred, zero_division=0)),
        "auc": float(auc),
        "latency_ms": float(latency_ms(model, Xte)),
        "n_params": int(n_params),
        "best_epoch": int(best_ep),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": int(len(te)),
    }
    with open(os.path.join(a.out_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"idx": te, "y_true": t_y, "p_anomaly": t_p}).to_csv(
        os.path.join(a.out_dir, f"{tag}_test_preds.csv"), index=False
    )
    print(f"TEST  acc {metrics['acc']:.3f}  F1 {metrics['f1']:.3f}  AUC {metrics['auc']:.3f}  "
          f"({metrics['latency_ms']:.2f} ms/ornek CPU)  [TN {tn} FP {fp} FN {fn} TP {tp}]")


if __name__ == "__main__":
    main()
