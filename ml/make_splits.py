"""C rolu: DOSYA bazinda train/val/test bolmesi uretir (splits.npz). Bir kez calistirilir, sabitlenir.

    python ml/make_splits.py --dir data

Tabaka (strata) = sinif + SIR + ornekleme hizi; boylece her bolmede her kosuldan dosya bulunur.
Ayni dosyanin tum pencereleri tek sette kalir (sizinti yok).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ft = pd.read_csv(os.path.join(a.dir, "file_index.csv"))
    fid = np.load(os.path.join(a.dir, "file_id.npy"))
    strata = (ft["sig"].astype(str) + "|sir" + ft["sir_db"].fillna(-99).astype(str)
              + "|fs" + ft["fs_hz"].astype(int).astype(str))
    if strata.value_counts().min() < 3:
        sys.exit("bazi tabakalarda 3'ten az dosya var; bolme guvenilir olmaz")

    ids = ft["file_id"].to_numpy()
    n_strata = strata.nunique()
    if len(ids) * min(a.val, a.test) < n_strata:
        sys.exit(f"{len(ids)} dosya, {n_strata} tabaka: val/test seti tabaka sayisindan kucuk kalir. "
                 "Daha cok dosya kullanin ya da --val/--test oranini arttirin.")
    rest, te = train_test_split(ids, test_size=a.test, stratify=strata, random_state=a.seed)
    st_rest = strata.loc[np.isin(ids, rest)]
    tr, va = train_test_split(rest, test_size=a.val / (1 - a.test), stratify=st_rest, random_state=a.seed)

    out = {f"{n}_idx": np.where(np.isin(fid, s))[0] for n, s in (("train", tr), ("val", va), ("test", te))}
    np.savez(os.path.join(a.dir, "splits.npz"), **out)

    y = np.load(os.path.join(a.dir, "y.npy"))
    print(f"{'set':6s} {'dosya':>6s} {'pencere':>8s} {'LTE':>6s} {'LTE+DSSS':>9s}")
    for n, s in (("train", tr), ("val", va), ("test", te)):
        idx = out[f"{n}_idx"]
        print(f"{n:6s} {len(s):6d} {len(idx):8d} {int((y[idx] == 0).sum()):6d} {int((y[idx] == 1).sum()):9d}")
    sir = ft.set_index("file_id")["sir_db"]
    print("test dosyalari, SIR dagilimi:", sir.loc[te].fillna(-1).astype(int).value_counts().sort_index().to_dict(), "(-1 = yalniz LTE)")


if __name__ == "__main__":
    main()
