from pathlib import Path
import os

_RF_ROOT = Path(os.environ.get("RF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))  # dsp/ klasoru
import numpy as np
import matplotlib.pyplot as plt

import sys

PROJECT_ROOT = _RF_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.io import load_iq


IQ_DIR = Path(os.environ.get("RF_IQ_DIR", "MATLAB_Dataset/IQ"))
OUT_DIR = (_RF_ROOT / "outputs" / "autocorrelation")

FS = 7_680_000
N_SAMPLES = 131072
MAX_LAG = 2000


def compute_autocorrelation(iq, max_lag):
    iq = iq[:N_SAMPLES]

    iq = iq - np.mean(iq)

    autocorr = np.correlate(
        iq,
        iq,
        mode="full"
    )

    center = len(autocorr) // 2

    autocorr = autocorr[
        center - max_lag:
        center + max_lag + 1
    ]

    autocorr = np.abs(autocorr)

    autocorr /= autocorr[max_lag] + 1e-12

    lags = np.arange(-max_lag, max_lag + 1)

    return lags, autocorr


def analyze_file(filename):
    iq = load_iq(IQ_DIR / filename)

    lags, autocorr = compute_autocorrelation(
        iq,
        MAX_LAG
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.plot(lags, autocorr)

    plt.xlabel("Lag (samples)")
    plt.ylabel("Normalized |Autocorrelation|")
    plt.title(f"Autocorrelation - {filename}")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = OUT_DIR / f"{filename}_autocorrelation.png"

    plt.savefig(output_path, dpi=150)

    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    analyze_file("OnlyLTE_frame_100")
    analyze_file("Combined_LTE_DSSS_frame_264")