from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\Users\asus\Desktop\RF-Anomaly-Detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.io import load_iq


IQ_DIR = Path(r"C:\Users\asus\Desktop\MATLAB_Dataset\IQ")
OUT_DIR = PROJECT_ROOT / "outputs" / "scf_v2new"

FS = 30_720_000

# AI pipeline'daki pencere
N_SAMPLES = 131072

# SCF segment
SEGMENT_LENGTH = 4096
OVERLAP = 2048
NFFT = 4096

# Cyclic frequency örnekleme sayısı
N_ALPHA = 65

# Spectral smoothing
SMOOTHING_BINS = 5


def load_signal(filename):
    iq = load_iq(IQ_DIR / filename)

    iq = iq[:N_SAMPLES]

    iq = iq - np.mean(iq)

    power = np.mean(np.abs(iq) ** 2)

    iq = iq / np.sqrt(power + 1e-12)

    return iq


def get_segments(iq):

    step = SEGMENT_LENGTH - OVERLAP

    n_segments = (
        (len(iq) - SEGMENT_LENGTH) // step
    ) + 1

    window = np.hanning(SEGMENT_LENGTH)

    segments = []

    for i in range(n_segments):

        start = i * step

        segment = iq[
            start:start + SEGMENT_LENGTH
        ]

        segment = segment * window

        segments.append(segment)

    return np.asarray(segments)


def compute_scf(iq):

    segments = get_segments(iq)

    spectra = np.fft.fft(
        segments,
        n=NFFT,
        axis=1
    )

    spectra = np.fft.fftshift(
        spectra,
        axes=1
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            NFFT,
            d=1 / FS
        )
    )

    df = FS / NFFT

    # α aralığı
    alpha_max = FS / 4

    alpha_values = np.linspace(
        -alpha_max,
        alpha_max,
        N_ALPHA
    )

    scf = np.zeros(
        (N_ALPHA, NFFT),
        dtype=np.complex128
    )

    coherence = np.zeros(
        (N_ALPHA, NFFT),
        dtype=np.float64
    )

    for a_idx, alpha in enumerate(alpha_values):

        # α/2 frekans kayması
        shift = int(
            round(
                (alpha / 2) / df
            )
        )

        if abs(shift) >= NFFT // 2:
            continue

        if shift == 0:

            x1 = spectra
            x2 = spectra

        elif shift > 0:

            x1 = spectra[:, shift:]
            x2 = spectra[:, :-shift]

        else:

            s = abs(shift)

            x1 = spectra[:, :-s]
            x2 = spectra[:, s:]

        length = min(
            x1.shape[1],
            x2.shape[1]
        )

        x1 = x1[:, :length]
        x2 = x2[:, :length]

        # Spectral correlation
        Sxy = np.mean(
            x1 * np.conj(x2),
            axis=0
        )

        P1 = np.mean(
            np.abs(x1) ** 2,
            axis=0
        )

        P2 = np.mean(
            np.abs(x2) ** 2,
            axis=0
        )

        # Normalize edilmiş spectral coherence
        C = np.abs(Sxy) ** 2 / (
            P1 * P2 + 1e-12
        )

        # Smoothing
        if SMOOTHING_BINS > 1:

            kernel = np.ones(
                SMOOTHING_BINS
            ) / SMOOTHING_BINS

            C = np.convolve(
                C,
                kernel,
                mode="same"
            )

        # Merkeze yerleştir
        if shift > 0:

            start = NFFT // 2
            end = min(
                start + length,
                NFFT
            )

            coherence[
                a_idx,
                start:end
            ] = C[
                :end - start
            ]

        elif shift < 0:

            end = NFFT // 2
            start = max(
                end - length,
                0
            )

            coherence[
                a_idx,
                start:end
            ] = C[
                -(end - start):
            ]

        else:

            start = NFFT // 2 - length // 2

            end = start + length

            coherence[
                a_idx,
                start:end
            ] = C[
                :end - start
            ]

    return frequencies, alpha_values, coherence


def save_plot(
    filename,
    frequencies,
    alpha_values,
    coherence
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    coherence_db = 10 * np.log10(
        coherence + 1e-12
    )

    # 0 dB = maximum coherence
    coherence_db -= np.max(
        coherence_db
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        coherence_db,
        origin="lower",
        aspect="auto",
        extent=[
            frequencies[0] / 1e6,
            frequencies[-1] / 1e6,
            alpha_values[0] / 1e6,
            alpha_values[-1] / 1e6
        ],
        cmap="viridis",
        vmin=-40,
        vmax=0
    )

    plt.colorbar(
        label="Normalized spectral coherence (dB)"
    )

    plt.xlabel(
        "Spectral frequency f (MHz)"
    )

    plt.ylabel(
        "Cyclic frequency α (MHz)"
    )

    plt.title(
        f"SCF / Spectral Coherence - {filename}"
    )

    plt.tight_layout()

    output = (
        OUT_DIR /
        f"{filename}_SCF_v2.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(f"Saved: {output}")


def analyze(filename):

    print(f"Analyzing: {filename}")

    iq = load_signal(filename)

    frequencies, alpha_values, coherence = compute_scf(
        iq
    )

    save_plot(
        filename,
        frequencies,
        alpha_values,
        coherence
    )


if __name__ == "__main__":

    analyze(
        "OnlyLTE_frame_100"
    )

    analyze(
        "Combined_LTE_DSSS_frame_264"
    )