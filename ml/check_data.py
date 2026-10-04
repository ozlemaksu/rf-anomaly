"""Veri sozlesmesi kontrolu. A ve C veriyi teslim etmeden once, B alip kullanmadan once calistirin.

Ornek:
    python ml/check_data.py --dir data
    python ml/check_data.py --dir data/fake

Beklenen dosyalar (--dir icinde):
    X_iq.npy      (N, 2, L)      float32   I ve Q kanallari          [A]
    X_spec.npy    (N, 1, F, T)   float32   dB genlik spektrogrami    [A]  (opsiyonel ama onerilir)
    y.npy         (N,)           int       0 = LTE, 1 = LTE+DSSS     [C]
    splits.npz    train_idx, val_idx, test_idx                       [C]
    file_id.npy   (N,)           int       pencerenin kaynak dosyasi [A]  (sizinti kontrolu icin sart)
    sir_db.npy    (N,)           float     opsiyonel, SIR'a gore analiz icin

Cikis kodu: 0 = hepsi tamam (UYARI olabilir), 1 = en az bir HATA var.
"""
import argparse
import os
import sys

import numpy as np

MIN_DIM = 16  # modeldeki 4 MaxPool(2) icin her boyut en az 16 olmali

results = []  # (durum, ad, ayrinti)


def report(status, name, detail=""):
    results.append((status, name, detail))
    tag = {"OK": "[ OK ]", "WARN": "[UYARI]", "FAIL": "[HATA]"}[status]
    print(f"{tag} {name}" + (f": {detail}" if detail else ""))


def need(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def check_array_basics(name, arr, ndim, chan=None):
    ok = True
    if arr.ndim != ndim:
        report("FAIL", f"{name} boyut sayisi", f"beklenen {ndim}, gelen {arr.ndim} (sekil {arr.shape})")
        return False
    report("OK", f"{name} sekli", str(arr.shape))
    if chan is not None and arr.shape[1] != chan:
        report("FAIL", f"{name} kanal sayisi", f"beklenen {chan}, gelen {arr.shape[1]}")
        ok = False
    if arr.dtype != np.float32:
        report("WARN", f"{name} veri tipi", f"{arr.dtype} (float32 onerilir; train.py cevirir)")
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        report("FAIL", f"{name} NaN/Inf", f"{n_bad} gecersiz deger")
        ok = False
    else:
        report("OK", f"{name} NaN/Inf yok")
    spatial = arr.shape[2:]
    if min(spatial) < MIN_DIM:
        report("FAIL", f"{name} uzamsal boyut", f"{spatial}: her boyut en az {MIN_DIM} olmali")
        ok = False
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data")
    a = p.parse_args()
    d = a.dir

    X_iq = need(os.path.join(d, "X_iq.npy"))
    X_spec = need(os.path.join(d, "X_spec.npy"))
    y = need(os.path.join(d, "y.npy"))
    file_id = need(os.path.join(d, "file_id.npy"))
    sir = need(os.path.join(d, "sir_db.npy"))
    sp_path = os.path.join(d, "splits.npz")
    sp = np.load(sp_path) if os.path.exists(sp_path) else None

    # --- var mi ---
    for name, obj in [("X_iq.npy", X_iq), ("y.npy", y), ("splits.npz", sp)]:
        if obj is None:
            report("FAIL", f"{name} bulunamadi", d)
    if X_spec is None:
        report("WARN", "X_spec.npy bulunamadi", "2D CNN icin gerekli")
    if file_id is None:
        report("WARN", "file_id.npy bulunamadi", "dosya bazli sizinti kontrolu yapilamayacak")
    if X_iq is None or y is None or sp is None:
        print("\nTemel dosyalar eksik, devam edilemiyor.")
        sys.exit(1)

    # --- sekiller ---
    check_array_basics("X_iq", X_iq, 3, chan=2)
    N = X_iq.shape[0]
    if X_spec is not None:
        check_array_basics("X_spec", X_spec, 4)
        if X_spec.shape[0] != N:
            report("FAIL", "X_spec ornek sayisi", f"{X_spec.shape[0]} != X_iq {N}")
        else:
            report("OK", "X_spec ve X_iq ayni ornek sayisinda")

    # --- etiketler ---
    if y.ndim != 1:
        report("FAIL", "y boyutu", f"(N,) beklenir, gelen {y.shape}")
    elif len(y) != N:
        report("FAIL", "y uzunlugu", f"{len(y)} != N {N}")
    else:
        vals = set(np.unique(y).tolist())
        if not vals <= {0, 1}:
            report("FAIL", "y degerleri", f"yalniz 0 ve 1 olmali, gelen {sorted(vals)}")
        else:
            frac = float(np.mean(y))
            report("OK", "y degerleri", f"anomali orani {frac:.2f} ({int(y.sum())} / {N})")
            if frac < 0.1 or frac > 0.9:
                report("WARN", "sinif dengesizligi", "agirlikli kayip kullaniliyor ama F1'e dikkat")

    # --- bolme ---
    keys = {"train_idx", "val_idx", "test_idx"}
    if not keys <= set(sp.files):
        report("FAIL", "splits.npz anahtarlari", f"gerekli {sorted(keys)}, gelen {sp.files}")
        sys.exit(1)
    idx = {k: np.asarray(sp[k]).astype(np.int64) for k in keys}
    in_range = all(len(v) > 0 and v.min() >= 0 and v.max() < N for v in idx.values())
    if not in_range:
        report("FAIL", "bolme indeksleri", f"bos set var veya [0, {N-1}] disinda indeks var")
    else:
        sizes = {k.replace("_idx", ""): len(v) for k, v in idx.items()}
        report("OK", "bolme boyutlari", str(sizes))
        tr, va, te = idx["train_idx"], idx["val_idx"], idx["test_idx"]
        overlaps = {
            "train-val": len(np.intersect1d(tr, va)),
            "train-test": len(np.intersect1d(tr, te)),
            "val-test": len(np.intersect1d(va, te)),
        }
        if any(overlaps.values()):
            report("FAIL", "bolmeler cakisiyor", str(overlaps))
        else:
            report("OK", "bolmeler birbirinden ayri")
        used = len(np.unique(np.r_[tr, va, te]))
        if used < N:
            report("WARN", "kullanilmayan ornekler", f"{N - used} ornek hicbir sette yok")
        if y.ndim == 1 and len(y) == N:
            for k, v in idx.items():
                c = np.bincount(y[v], minlength=2)
                if c.min() == 0:
                    report("FAIL", f"{k} tek sinif iceriyor", f"LTE {c[0]}, LTE+DSSS {c[1]}")
                else:
                    report("OK", f"{k} sinif dagilimi", f"LTE {c[0]}, LTE+DSSS {c[1]}")

        # --- sizinti: dosya bazli ---
        if file_id is not None:
            if len(file_id) != N:
                report("FAIL", "file_id uzunlugu", f"{len(file_id)} != N {N}")
            else:
                f_tr, f_va, f_te = (set(file_id[v].tolist()) for v in (tr, va, te))
                leaks = {
                    "train-val": len(f_tr & f_va),
                    "train-test": len(f_tr & f_te),
                    "val-test": len(f_va & f_te),
                }
                if any(leaks.values()):
                    report("FAIL", "DOSYA SIZINTISI", f"ayni dosyadan pencereler farkli setlerde: {leaks}")
                else:
                    report("OK", "dosya sizintisi yok", f"{len(f_tr)}/{len(f_va)}/{len(f_te)} dosya")
                if y.ndim == 1 and len(y) == N:
                    mixed = [f for f in np.unique(file_id) if len(np.unique(y[file_id == f])) > 1]
                    if mixed:
                        report("FAIL", "bir dosyada hem LTE hem LTE+DSSS etiketi", f"ilk dosyalar: {mixed[:5]}")

        # --- sizinti: birebir ayni pencere ---
        def row_hashes(v):
            return {hash(X_iq[i].tobytes()) for i in v}

        h_tr, h_te = row_hashes(tr), row_hashes(te)
        dup = len(h_tr & h_te)
        if dup:
            report("FAIL", "birebir ayni pencere", f"train ve test'te {dup} ortak pencere var")
        else:
            report("OK", "train/test'te birebir ayni pencere yok")

    # --- olcek ---
    if np.isfinite(X_iq).all():
        ch_std = X_iq.std(axis=(0, 2))
        ch_mean = X_iq.mean(axis=(0, 2))
        report("OK", "IQ olcegi", f"ortalama {np.round(ch_mean, 4).tolist()}, std {np.round(ch_std, 4).tolist()}")
        if (ch_std == 0).any():
            report("FAIL", "bir IQ kanali sabit", "std = 0")
        elif ch_std.max() > 1e4 or ch_std.min() < 1e-6:
            report("WARN", "IQ olcegi uc degerde", "int16 ham sayilar veya yanlis dtype olabilir")
    if X_spec is not None and np.isfinite(X_spec).all():
        report("OK", "spektrogram araligi", f"min {X_spec.min():.1f}, max {X_spec.max():.1f}")
    if sir is not None and len(sir) == N:
        valid = sir[~np.isnan(sir)]
        if len(valid):
            report("OK", "sir_db", f"{valid.min():.1f} .. {valid.max():.1f} dB")

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in results if s == "WARN")
    print(f"\nOZET: {len(results) - n_fail - n_warn} tamam, {n_warn} uyari, {n_fail} hata")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
