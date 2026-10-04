"""Sahte veri uretici: gercek veri gelene kadar boru hattini denemek icin.

LTE benzeri (OFDM) sinyal + istege bagli zayif DSSS (anomali) uretir.
UYARI: Bu veri YALNIZCA kodu sinamak icindir. Buradan cikan dogruluk
rakamlari bilimsel sonuc degildir, sunumda kullanilmaz.

Ornek:
    python ml/make_fake_data.py --out data/fake
"""
import argparse
import os

import numpy as np
from scipy.signal import stft


def lte_like(rng, length, n_fft=256, n_used=150, cp=18):
    """OFDM benzeri sinyal (QPSK alt tasiyicilar + dongusel onek)."""
    sym_len = n_fft + cp
    n_sym = length // sym_len + 2
    idx = np.r_[1 : n_used // 2 + 1, n_fft - n_used // 2 : n_fft]
    out = []
    for _ in range(n_sym):
        grid = np.zeros(n_fft, dtype=np.complex128)
        bits = rng.integers(0, 4, n_used)
        grid[idx] = np.exp(1j * (np.pi / 4 + np.pi / 2 * bits))
        t = np.fft.ifft(grid) * np.sqrt(n_fft)
        out.append(np.r_[t[-cp:], t])
    x = np.concatenate(out)
    off = int(rng.integers(0, sym_len))
    return x[off : off + length]


def dsss_like(rng, length, code, f0, sps=2):
    """PN kodu ile yayilmis BPSK (DSSS benzeri), f0 kadar frekans kaydirilmis."""
    chips_per_bit = len(code)
    n_bits = length // (chips_per_bit * sps) + 2
    bits = rng.choice([-1.0, 1.0], n_bits)
    chips = (bits[:, None] * code[None, :]).ravel()
    x = np.repeat(chips, sps)[:length].astype(np.complex128)
    return x * np.exp(2j * np.pi * f0 * np.arange(length))


def rms(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))


def make_file(rng, n_windows, length, anomaly, snr_db, sir_db, gain_jitter_db):
    code = rng.choice([-1.0, 1.0], 31)
    f0 = rng.uniform(-0.15, 0.15)
    wins = []
    for _ in range(n_windows):
        x = lte_like(rng, length)
        x = x / rms(x)
        if anomaly:
            d = dsss_like(rng, length, code, f0)
            d = d / rms(d) * 10 ** (-sir_db / 20)
            x = x + d
        noise = (rng.normal(size=length) + 1j * rng.normal(size=length)) * np.sqrt(
            10 ** (-snr_db / 10) / 2
        )
        x = x + noise
        x = x * 10 ** (rng.uniform(-gain_jitter_db, gain_jitter_db) / 20)
        wins.append(x)
    return wins


def to_spec(x, nperseg=64):
    """Karmasik IQ -> dB genlik spektrogrami, (F, T), frekans merkezli."""
    _, _, z = stft(x, nperseg=nperseg, noverlap=nperseg // 2, return_onesided=False)
    z = np.fft.fftshift(z, axes=0)
    return (20 * np.log10(np.abs(z) + 1e-12)).astype(np.float32)


def stratified_file_split(rng, file_labels, frac=(0.7, 0.15, 0.15)):
    """Dosya bazinda, siniflari koruyarak bol. Donus: 3 dosya-indeksi dizisi."""
    parts = [[], [], []]
    for lab in (0, 1):
        files = np.where(file_labels == lab)[0]
        rng.shuffle(files)
        n = len(files)
        n_tr = max(1, int(round(frac[0] * n)))
        n_va = max(1, int(round(frac[1] * n)))
        parts[0] += list(files[:n_tr])
        parts[1] += list(files[n_tr : n_tr + n_va])
        parts[2] += list(files[n_tr + n_va :])
    return [np.array(sorted(p)) for p in parts]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fake")
    p.add_argument("--n_files", type=int, default=60)
    p.add_argument("--windows_per_file", type=int, default=10)
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--snr_db", type=float, nargs=2, default=[15, 25])
    p.add_argument("--sir_db", type=float, nargs=2, default=[5, 25])
    p.add_argument("--gain_jitter_db", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    iq, spec, y, file_id, sir = [], [], [], [], []
    file_labels = np.zeros(a.n_files, dtype=int)
    for f in range(a.n_files):
        # ICARUS'taki kural: tek dongu indeksi = LTE+DSSS, cift = yalniz LTE
        anomaly = f % 2 == 1
        file_labels[f] = int(anomaly)
        snr = rng.uniform(*a.snr_db)
        sr = rng.uniform(*a.sir_db)
        wins = make_file(rng, a.windows_per_file, a.length, anomaly, snr, sr, a.gain_jitter_db)
        for w in wins:
            iq.append(np.stack([w.real, w.imag]).astype(np.float32))
            spec.append(to_spec(w)[None])
            y.append(int(anomaly))
            file_id.append(f)
            sir.append(sr if anomaly else np.nan)

    X_iq = np.stack(iq).astype(np.float32)
    X_spec = np.stack(spec).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    file_id = np.array(file_id, dtype=np.int64)
    sir = np.array(sir, dtype=np.float32)

    tr_f, va_f, te_f = stratified_file_split(rng, file_labels)
    tr = np.where(np.isin(file_id, tr_f))[0]
    va = np.where(np.isin(file_id, va_f))[0]
    te = np.where(np.isin(file_id, te_f))[0]

    np.save(os.path.join(a.out, "X_iq.npy"), X_iq)
    np.save(os.path.join(a.out, "X_spec.npy"), X_spec)
    np.save(os.path.join(a.out, "y.npy"), y)
    np.save(os.path.join(a.out, "file_id.npy"), file_id)
    np.save(os.path.join(a.out, "sir_db.npy"), sir)
    np.savez(os.path.join(a.out, "splits.npz"), train_idx=tr, val_idx=va, test_idx=te)

    print(f"yazildi: {a.out}")
    print(f"  X_iq   {X_iq.shape} {X_iq.dtype}")
    print(f"  X_spec {X_spec.shape} {X_spec.dtype}")
    print(f"  y      {y.shape}  anomali orani {y.mean():.2f}")
    print(f"  bolme  train {len(tr)}  val {len(va)}  test {len(te)}  (dosya bazinda)")


if __name__ == "__main__":
    main()
