"""A rolu: ICARUS Synthetic (MATLAB_Dataset) ham dosyalarindan egitim verisini uretir.

Calistirma (repo kokunden):
    python src/preprocess.py --root data/ham/sentetik/MATLAB_Dataset --out data

Dosya formati (gercek veride dogrulandi):
    8 bayt baslik = iki int32 [2, N]   (N = karmasik ornek sayisi)
    ardindan 2*N adet float32 little-endian, IC ICE:  I0, Q0, I1, Q1, ...
Her IQ dosyasinin (uzantisiz) ayni adli bir Metadata/<ad>.csv dosyasi var.
Dosyalar farkli ornekleme hizinda (7.68 / 15.36 / 30.72 MHz, hepsi 80 ms), bu yuzden
her dosyadan ESIT sayida (--wpf) pencere alinir; aksi halde 30.72 MHz dosyalari baskin olur.

Uretilen dosyalar (--out):
    X_iq.npy        (N, 2, L)   float32   I ve Q
    X_spec.npy      (N, 1, F, T) float32  STFT guc spektrogrami (dB), frekans ekseni fftshift'li
    file_id.npy     (N,)        int32     pencerenin kayit dosyasi (file_index.csv satiri)
    y.npy           (N,)        int64     0 = yalniz LTE, 1 = LTE + DSSS
    sir_db.npy      (N,)        float32   LTE-only icin NaN
    fs_hz.npy       (N,)        float32   ornekleme hizi (analiz icin, modele girmez)
    file_index.csv              file_id, file, sig, y, fs_hz, snr_db, sir_db
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import stft


def read_iq(path):
    """Dosyayi okur, (I, Q) dondurur (float32)."""
    with open(path, "rb") as f:
        b = f.read()
    head = np.frombuffer(b, dtype="<i4", count=2)
    if head[0] != 2:
        raise ValueError(f"{path}: beklenmeyen baslik {head.tolist()} (ilk deger 2 olmali)")
    n = int(head[1])
    a = np.frombuffer(b, dtype="<f4", count=2 * n, offset=8)
    if a.size != 2 * n:
        raise ValueError(f"{path}: baslik {n} ornek diyor ama dosya kisa")
    return a[0::2], a[1::2]


def spectrogram_db(i, q, nfft, hop):
    """Karmasik IQ'dan (F, T) guc spektrogrami (dB). Iki tarafli, fftshift'li."""
    x = (i.astype(np.float32) + 1j * q.astype(np.float32)).astype(np.complex64)
    _, _, z = stft(x, window="hann", nperseg=nfft, noverlap=nfft - hop,
                   return_onesided=False, boundary=None, padded=False)
    z = np.fft.fftshift(z, axes=0)
    return (10.0 * np.log10(np.abs(z) ** 2 + 1e-12)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="IQ/ ve Metadata/ klasorlerini iceren klasor")
    p.add_argument("--out", default="data")
    p.add_argument("--L", type=int, default=131072, help="pencere uzunlugu (ornek)")
    p.add_argument("--wpf", type=int, default=4, help="dosya basina pencere sayisi")
    p.add_argument("--nfft", type=int, default=256)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="yalnizca ilk N dosya (hizli deneme)")
    a = p.parse_args()

    iqd, mdd = os.path.join(a.root, "IQ"), os.path.join(a.root, "Metadata")
    for d in (iqd, mdd):
        if not os.path.isdir(d):
            sys.exit(f"{d} bulunamadi. --root yolunu kontrol edin.")
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for f in sorted(os.listdir(iqd)):
        c = os.path.join(mdd, f + ".csv")
        if not os.path.exists(c):
            print(f"[UYARI] {f}: Metadata csv yok, atlandi")
            continue
        m = pd.read_csv(c).iloc[0]
        sig = str(m["Signal_Type"]).strip()
        if sig not in ("LTE", "LTE_DSSS"):
            sys.exit(f"{f}: beklenmeyen Signal_Type '{sig}'")
        rows.append(dict(
            file=f, sig=sig, y=int(sig == "LTE_DSSS"),
            fs_hz=float(m["LTE_SR"]),
            snr_db=float(pd.to_numeric(m.get("LTE_SNR_dB"), errors="coerce")),
            sir_db=float(pd.to_numeric(m.get("LTE_DSSS_SIR_dB"), errors="coerce")) if sig == "LTE_DSSS" else np.nan,
        ))
    if a.limit:
        rows = rows[: a.limit]
    nfile = len(rows)
    if nfile == 0:
        sys.exit("hic dosya yok")

    ftab = pd.DataFrame(rows)
    ftab.insert(0, "file_id", np.arange(nfile))

    probe = spectrogram_db(np.zeros(a.L, np.float32), np.zeros(a.L, np.float32), a.nfft, a.hop)
    F, T = probe.shape
    if min(a.L, F, T) < 16:
        sys.exit(f"L={a.L}, F={F}, T={T}: hepsi >= 16 olmali (nfft/hop'u kucultun)")
    N = nfile * a.wpf
    gb = (N * 2 * a.L + N * F * T) * 4 / 1e9
    print(f"{nfile} dosya x {a.wpf} pencere = {N} pencere; X_iq (N,2,{a.L}), X_spec (N,1,{F},{T}); ~{gb:.2f} GB")

    X_iq = np.lib.format.open_memmap(os.path.join(a.out, "X_iq.npy"), mode="w+",
                                     dtype=np.float32, shape=(N, 2, a.L))
    X_sp = np.lib.format.open_memmap(os.path.join(a.out, "X_spec.npy"), mode="w+",
                                     dtype=np.float32, shape=(N, 1, F, T))
    file_id = np.zeros(N, np.int32)
    y = np.zeros(N, np.int64)
    sir = np.full(N, np.nan, np.float32)
    fs = np.zeros(N, np.float32)

    for k, r in ftab.iterrows():
        i, q = read_iq(os.path.join(iqd, r["file"]))
        n = len(i)
        if n < a.L:
            sys.exit(f"{r['file']}: {n} ornek var, L={a.L} icin yetersiz")
        starts = np.linspace(0, n - a.L, a.wpf).astype(int) if a.wpf > 1 else np.array([0])
        if a.wpf > 1 and (n - a.L) / (a.wpf - 1) < a.L:
            print(f"[UYARI] {r['file']}: pencereler cakisiyor (n={n})")
        for j, s in enumerate(starts):
            idx = k * a.wpf + j
            wi, wq = i[s:s + a.L], q[s:s + a.L]
            X_iq[idx, 0], X_iq[idx, 1] = wi, wq
            X_sp[idx, 0] = spectrogram_db(wi, wq, a.nfft, a.hop)
            file_id[idx], y[idx], sir[idx], fs[idx] = k, r["y"], r["sir_db"], r["fs_hz"]
        if (k + 1) % 50 == 0 or k + 1 == nfile:
            print(f"  {k + 1}/{nfile} dosya islendi", flush=True)

    X_iq.flush()
    X_sp.flush()
    np.save(os.path.join(a.out, "file_id.npy"), file_id)
    np.save(os.path.join(a.out, "y.npy"), y)
    np.save(os.path.join(a.out, "sir_db.npy"), sir)
    np.save(os.path.join(a.out, "fs_hz.npy"), fs)
    ftab.to_csv(os.path.join(a.out, "file_index.csv"), index=False)
    print(f"Bitti. LTE penceresi {int((y == 0).sum())}, LTE+DSSS penceresi {int((y == 1).sum())}.")
    print("Sonraki adim: python src/make_splits.py --dir", a.out)


if __name__ == "__main__":
    main()
