from pathlib import Path
import os

_RF_ROOT = Path(os.environ.get("RF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))  # dsp/ klasoru

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft


IQ_DIR = Path(os.environ.get("RF_IQ_DIR", "MATLAB_Dataset/IQ"))
NFFT = 128

OUT_DIR = (_RF_ROOT / "outputs" / "window_visualization") / f"nfft_{NFFT}"

FILE_NAME = "Combined_LTE_DSSS_frame_264"

WINDOW_LENGTH = 131072
N_WINDOWS = 4

NFFT = 512
HOP = 256


def find_iq(iq_dir, name):
    matches = list(iq_dir.glob(name)) + list(iq_dir.glob(name + ".*"))
    return matches[0] if matches else None


def read_iq(path):
    with open(path, "rb") as f:
        f.seek(8)
        raw = np.fromfile(f, dtype=np.float32)

    i = raw[0::2]
    q = raw[1::2]

    return i + 1j * q


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    path = find_iq(IQ_DIR, FILE_NAME)

    if path is None:
        raise FileNotFoundError(f"IQ file bulunamadı: {FILE_NAME}")

    iq = read_iq(path)

    # Bu recording için metadata'dan bilinen sample rate
    fs = 30.72e6

    if len(iq) < WINDOW_LENGTH:
        raise ValueError("IQ recording 131072 örnekten kısa.")

    starts = np.linspace(
        0,
        len(iq) - WINDOW_LENGTH,
        N_WINDOWS,
        dtype=int,
    )

    # Önce bütün window'ların spectrogramlarını hesapla.
    results = []

    global_min = np.inf
    global_max = -np.inf

    for w, start in enumerate(starts):

        segment = iq[start:start + WINDOW_LENGTH]

        f, t, Z = stft(
            segment,
            fs=fs,
            window="hann",
            nperseg=NFFT,
            noverlap=0,
            nfft=NFFT,
            boundary=None,
            padded=False,
            return_onesided=False,
        )

        f = np.fft.fftshift(f)
        Z = np.fft.fftshift(Z, axes=0)

        power_db = 20 * np.log10(np.abs(Z) + 1e-12)

        global_min = min(global_min, power_db.min())
        global_max = max(global_max, power_db.max())

        results.append((w, start, f, t, power_db))

    # Aynı renk ölçeğini bütün window'larda kullan.
    vmin = global_min
    vmax = global_max

    for w, start, f, t, power_db in results:

        fig, ax = plt.subplots(figsize=(10, 5))

        mesh = ax.pcolormesh(
            t * 1000,
            f / 1e6,
            power_db,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(
            f"{FILE_NAME} — Window {w}\n"
            f"start sample = {start}"
        )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")

        fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")

        fig.tight_layout()

        output = OUT_DIR / f"{FILE_NAME}_window_{w}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)

        print(f"Saved: {output}")

    print("\nTamamlandı.")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()