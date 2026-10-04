import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft, welch
from scipy.stats import spearmanr

L = 131072
WPF = 4


def read_iq(path):
    b = open(path, "rb").read()
    head = np.frombuffer(b, dtype="<i4", count=2)
    if head[0] != 2:
        raise ValueError(f"{path}: beklenmeyen baslik {head.tolist()}")
    n = int(head[1])
    a = np.frombuffer(b, dtype="<f4", count=2 * n, offset=8)
    return a[0::2], a[1::2]


def find_iq(iq_dir, name):
    p = os.path.join(iq_dir, name)
    if os.path.exists(p):
        return p
    hits = [h for h in glob.glob(os.path.join(iq_dir, "*" + name)) if os.path.basename(h).endswith(name)]
    return hits[0] if hits else None


def meta_bw(meta_dir, name, bw_col=None):
    """Metadata CSV'den nominal LTE bant genisligi (Hz). Bulunamazsa (None, None)."""
    if not meta_dir:
        return None, None
    p = os.path.join(meta_dir, name + ".csv")
    if not os.path.exists(p):
        return None, None
    m = pd.read_csv(p).iloc[0]
    cols = [bw_col] if bw_col else [c for c in m.index if re.search(r"bw|bandwidth|band_width", str(c), re.I)]
    for c in cols:
        v = pd.to_numeric(m.get(c), errors="coerce")
        if pd.notna(v) and v > 0:
            return (float(v) * 1e6 if v < 1000 else float(v)), f"metadata:{c}"
    return None, None


def estimate_bw(i, q, fs):
    """Kaydin ortalama PSD'sinden kapsanan bant genisligi (Hz): plato ile gurultu tabaninin ortasi."""
    z = i.astype(np.float32) + 1j * q.astype(np.float32)
    f, p = welch(z, fs=fs, nperseg=4096, return_onesided=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    pdb = 10 * np.log10(np.convolve(p, np.ones(31) / 31, mode="same") + 1e-30)
    thr = (np.percentile(pdb, 90) + np.percentile(pdb, 10)) / 2
    c = len(f) // 2
    lo = hi = c
    while lo > 0 and pdb[lo - 1] > thr:
        lo -= 1
    while hi < len(f) - 1 and pdb[hi + 1] > thr:
        hi += 1
    return float(f[hi] - f[lo])


def window_power(i, q, fs, s, nfft, hop):
    x = (i[s:s + L].astype(np.float32) + 1j * q[s:s + L].astype(np.float32)).astype(np.complex64)
    f, _, Z = stft(x, fs=fs, window="hann", nperseg=nfft, noverlap=nfft - hop,
                   return_onesided=False, boundary=None, padded=False)
    return np.fft.fftshift(f), np.fft.fftshift(np.abs(Z) ** 2, axes=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq-dir", required=True)
    ap.add_argument("--meta-dir", default=None)
    ap.add_argument("--pred", required=True, help="annotated_predictions.csv")
    ap.add_argument("--out", default="band_ratio_out")
    ap.add_argument("--nfft", type=int, default=256)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--bw-col", default=None, help="Metadata'daki bant genisligi sutununun adi (otomatik bulunamazsa)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.pred)
    rows, bw_log, missing = [], {}, []
    for name, g in df.groupby("file"):
        path = find_iq(a.iq_dir, name)
        if path is None:
            missing.append(name)
            continue
        i, q = read_iq(path)
        n = len(i)
        fs = float(g.fs_hz.iloc[0])
        nominal, src = meta_bw(a.meta_dir, name, a.bw_col)
        if nominal:
            bw_occ = 0.9 * nominal
        else:
            bw_occ, src = estimate_bw(i, q, fs), "sinyal"
        bw_log[name] = (fs / 1e6, bw_occ / 1e6, src)
        starts = np.linspace(0, n - L, WPF).astype(int)
        for _, r in g.iterrows():
            f, P = window_power(i, q, fs, starts[int(r.window_in_file)], a.nfft, a.hop)
            inb = np.abs(f) <= 0.8 * bw_occ / 2
            oob = np.abs(f) >= 1.2 * bw_occ / 2
            ratio = 10 * np.log10(P[inb].mean() / P[oob].mean())
            fr = 10 * np.log10(P[inb].mean(axis=0) / P[oob].mean(axis=0))   # kolon (zaman) bazinda oran
            rows.append(dict(idx=r.idx, file=name, window_in_file=int(r.window_in_file), fs_hz=r.fs_hz,
                             snr_db=r.snr_db, sir_db=r.sir_db, y_true=r.y_true, cnn2d_prob=r.cnn2d_prob,
                             cnn2d_error=r.cnn2d_error, band_ratio_db=round(float(ratio), 3),
                             busy_frac=round(float((fr > 2.0).mean()), 3), bw_occ_mhz=round(bw_occ / 1e6, 2),
                             bw_kaynak=src))
    if missing:
        print(f"[UYARI] IQ dosyasi bulunamadi, atlandi: {len(missing)} recording")
    d = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    d.to_csv(os.path.join(a.out, "band_ratio_table.csv"), index=False)
    print(f"{d.file.nunique()} recording, {len(d)} pencere / {len(df)}")
    print("\nrecording | fs MHz | BW_occ MHz | kaynak")
    for k, (fs_, bw_, s_) in sorted(bw_log.items()):
        print(f"  {k:30s} {fs_:6.2f} {bw_:7.2f}  {s_}")

    d["durum"] = np.where(d.cnn2d_error == "dogru", "dogru", d.cnn2d_error)
    col = {"dogru": "0.65", "FP": "tab:red", "FN": "tab:blue"}
    mk = {7.68e6: "o", 15.36e6: "s", 30.72e6: "^"}

    # --- 1) scatter: band_ratio vs cnn2d_prob (FP/FN ayri renk, fs ayri isaret) ---
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for st in ["dogru", "FP", "FN"]:
        for fs_, m in mk.items():
            s = d[(d.durum == st) & (d.fs_hz == fs_)]
            if len(s):
                ax.scatter(s.band_ratio_db, s.cnn2d_prob, c=col[st], marker=m, s=40 if st == "dogru" else 70,
                           edgecolors="k" if st != "dogru" else "none", linewidths=.6,
                           label=f"{st}, {fs_/1e6:g} MHz")
    ax.axhline(.5, color="k", ls="--", lw=.8)
    ax.set_xlabel("band_ratio_db"); ax.set_ylabel("cnn2d_prob")
    ax.set_title(f"band_ratio vs cnn2d_prob (n={len(d)}), Spearman={spearmanr(d.band_ratio_db, d.cnn2d_prob)[0]:.2f}")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "1_scatter_ratio_vs_prob.png"), dpi=130); plt.close(fig)

    # --- 2) FP / FN ayri: SNR x sinif panelleri ---
    snrs = sorted(d.snr_db.unique())
    fig, axs = plt.subplots(2, len(snrs), figsize=(4.2 * len(snrs), 7), squeeze=False, sharey=True)
    for r_, y_ in enumerate([0, 1]):
        for c_, sn in enumerate(snrs):
            ax = axs[r_, c_]; s0 = d[(d.y_true == y_) & (d.snr_db == sn)]
            for st in ["dogru", "FP", "FN"]:
                for fs_, m in mk.items():
                    s = s0[(s0.durum == st) & (s0.fs_hz == fs_)]
                    if len(s):
                        ax.scatter(s.band_ratio_db, s.cnn2d_prob, c=col[st], marker=m, s=45,
                                   edgecolors="k" if st != "dogru" else "none", linewidths=.6)
            ax.axhline(.5, color="k", ls="--", lw=.8); ax.grid(alpha=.3)
            rho = spearmanr(s0.band_ratio_db, s0.cnn2d_prob)[0] if len(s0) > 4 else np.nan
            ax.set_title(f"{'OnlyLTE' if y_ == 0 else 'LTE+DSSS'} | SNR {sn:g} dB | n={len(s0)} | rho={rho:.2f}", fontsize=9)
            ax.set_xlabel("band_ratio_db")
            if c_ == 0: ax.set_ylabel("cnn2d_prob")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "2_scatter_by_snr_class.png"), dpi=130); plt.close(fig)

    # --- 3) fs bazinda dagilim ---
    def strip(ax, s0, xcol, xs, xlabel):
        for k, xv in enumerate(xs):
            for st in ["dogru", "FP", "FN"]:
                s = s0[(s0[xcol].fillna(-1) == xv) & (s0.durum == st)]
                jit = (np.random.RandomState(k).rand(len(s)) - .5) * .35 + {"dogru": -.12, "FP": .12, "FN": .12}[st]
                ax.scatter(k + jit, s.band_ratio_db, c=col[st], s=28, edgecolors="k" if st != "dogru" else "none", linewidths=.4)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.set_xlabel(xlabel)
        ax.set_ylabel("band_ratio_db"); ax.grid(alpha=.3)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fss = sorted(d.fs_hz.unique() / 1e6)
    for ax, y_ in zip(axs, [0, 1]):
        s0 = d[d.y_true == y_].assign(fsm=lambda t: t.fs_hz / 1e6)
        strip(ax, s0, "fsm", fss, "fs (MHz)"); ax.set_title("OnlyLTE" if y_ == 0 else "LTE+DSSS")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "3_ratio_by_fs.png"), dpi=130); plt.close(fig)

    # --- 4) SNR / SIR bazinda dagilim ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    strip(axs[0], d[d.y_true == 0], "snr_db", sorted(d[d.y_true == 0].snr_db.unique()), "SNR (dB) - OnlyLTE")
    strip(axs[1], d[d.y_true == 1], "snr_db", sorted(d[d.y_true == 1].snr_db.unique()), "SNR (dB) - LTE+DSSS")
    strip(axs[2], d[d.y_true == 1], "sir_db", sorted(d[d.y_true == 1].sir_db.dropna().unique()), "SIR (dB) - LTE+DSSS")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "4_ratio_by_snr_sir.png"), dpi=130); plt.close(fig)

    # --- ozet istatistik ---
    pd.set_option("display.width", 200)
    print("\nGrup ozeti (sinif, SNR, SIR, durum): n, ort. band_ratio_db, ort. cnn2d_prob")
    print(d.groupby(["y_true", "snr_db", d.sir_db.fillna(-1), "durum"]).agg(
        n=("idx", "size"), ratio=("band_ratio_db", "mean"), p2d=("cnn2d_prob", "mean")).round(3).to_string())
    print("\nSpearman cnn2d_prob ~ band_ratio_db (n>=8 gruplar):")
    for (y_, sn), g in d.groupby(["y_true", "snr_db"]):
        if len(g) >= 8:
            print(f"  y={y_} SNR={sn:g}: n={len(g)} rho={spearmanr(g.band_ratio_db, g.cnn2d_prob)[0]:.2f}  "
                  f"(busy_frac: {spearmanr(g.busy_frac, g.cnn2d_prob)[0]:.2f})")


if __name__ == "__main__":
    main()
