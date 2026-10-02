import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from models import CNN1D, CNN2D


def make_loader(X, y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, loss_fn, dev):
    model.eval()
    total, probs, ys = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        out = model(xb)
        total += loss_fn(out, yb).item() * len(yb)
        probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        ys.append(yb.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(probs), np.concatenate(ys)


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
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.load(a.x).astype(np.float32)
    y = np.load(a.y).astype(np.int64)
    sp = np.load(a.splits)
    tr, va, te = sp["train_idx"], sp["val_idx"], sp["test_idx"]

    # Normalizasyon istatistiği yalnızca eğitim setinden (sızıntıyı önler)
    axes = (0, 2) if a.model == "cnn1d" else (0, 2, 3)
    mu = X[tr].mean(axis=axes, keepdims=True)
    sd = X[tr].std(axis=axes, keepdims=True) + 1e-8
    X = (X - mu) / sd

    model = (CNN1D() if a.model == "cnn1d" else CNN2D(in_ch=X.shape[1])).to(dev)

    counts = np.bincount(y[tr], minlength=2)
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    tr_ld = make_loader(X[tr], y[tr], a.bs, True)
    va_ld = make_loader(X[va], y[va], a.bs, False)
    te_ld = make_loader(X[te], y[te], a.bs, False)

    best, wait = float("inf"), 0
    ckpt = f"results/{a.model}_best.pt"
    for ep in range(a.epochs):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        v_loss, v_p, v_y = evaluate(model, va_ld, loss_fn, dev)
        v_f1 = f1_score(v_y, v_p > 0.5)
        print(f"ep {ep:02d}  val_loss {v_loss:.4f}  val_F1 {v_f1:.3f}")
        if v_loss < best:
            best, wait = v_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= a.patience:
                print("erken durdurma")
                break

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    _, t_p, t_y = evaluate(model, te_ld, loss_fn, dev)
    print(f"TEST  acc {((t_p > 0.5) == t_y).mean():.3f}  F1 {f1_score(t_y, t_p > 0.5):.3f}")

    # C kişisi metrikleri bu dosyadan hesaplayacak
    pd.DataFrame({"idx": te, "y_true": t_y, "p_anomaly": t_p}).to_csv(
        f"results/{a.model}_test_preds.csv", index=False
    )


if __name__ == "__main__":
    main()