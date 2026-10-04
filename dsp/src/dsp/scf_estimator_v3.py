from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\Users\asus\Desktop\RF-Anomaly-Detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.io import load_iq


IQ_DIR = Path(r"C:\Users\asus\Desktop\MATLAB_Dataset\IQ")
OUT_DIR = PROJECT_ROOT / "outputs" / "scf_v3"

FS = 30_720_000
N_SAMPLES = 131072

SEGMENT_LENGTH = 4096
OVERLAP = 2048
NFFT = 4096

# α/2 için FFT bin kaymaları
MAX_SHIFT = 512

SMOOTHING_BINS = 5


def load_signal(filename):

    iq = load_iq(IQ_DIR / filename)

    iq = iq[:N_SAMPLES]

    iq = iq - np.mean(iq)

    rms = np.sqrt(np.mean(np.abs(iq) ** 2))

    iq = iq / (rms + 1e-12)

    return iq


def create_segments(iq):

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

        segments.append(
            segment * window
        )

    return np.asarray(segments)


def compute_scf(iq):

    segments = create_segments(iq)

    X = np.fft.fft(
        segments,
        n=NFFT,
        axis=1
    )

    X = np.fft.fftshift(
        X,
        axes=1
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            NFFT,
            d=1 / FS
        )
    )

    df = FS / NFFT

    coherence = np.full(
        (MAX_SHIFT + 1, NFFT),
        np.nan
    )

    alpha_values = (
        2 * np.arange(MAX_SHIFT + 1) * df
    )

    # α = 0
    S0 = np.mean(
        np.abs(X) ** 2,
        axis=0
    )

    coherence[0, :] = 1.0

    for m in range(1, MAX_SHIFT + 1):

        # α/2 = m * df
        #
        # f + α/2  -> index j + m
        # f - α/2  -> index j - m

        length = NFFT - 2 * m

        X_plus = X[:, 2 * m:]
        X_minus = X[:, :length]

        S_alpha = np.mean(
            X_plus * np.conj(X_minus),
            axis=0
        )

        P_plus = np.mean(
            np.abs(X_plus) ** 2,
            axis=0
        )

        P_minus = np.mean(
            np.abs(X_minus) ** 2,
            axis=0
        )

        # Spectral coherence
        C = (
            np.abs(S_alpha) ** 2
            /
            (
                P_plus * P_minus
                + 1e-12
            )
        )

        if SMOOTHING_BINS > 1:

            kernel = (
                np.ones(SMOOTHING_BINS)
                / SMOOTHING_BINS
            )

            C = np.convolve(
                C,
                kernel,
                mode="same"
            )

        # Sayısal taşmaları önle
        C = np.clip(C, 0.0, 1.0)

        # C'nin ait olduğu f indeksleri:
        # j = m ... NFFT-m-1

        coherence[
            m,
            m:NFFT - m
        ] = C

    return (
        frequencies,
        alpha_values,
        coherence
    )


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

    # α=0 satırını feature analizinden ayrı tutuyoruz.
    plot_data = coherence.copy()

    plot_data[0, :] = np.nan

    coherence_db = (
        10
        * np.log10(
            plot_data + 1e-12
        )
    )

    plt.figure(
        figsize=(13, 8)
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
        label="Spectral coherence (dB)"
    )

    plt.xlabel(
        "Spectral frequency f (MHz)"
    )

    plt.ylabel(
        "Cyclic frequency α (MHz)"
    )

    plt.title(
        f"SCF V3 - {filename}"
    )

    plt.tight_layout()

    output = (
        OUT_DIR
        / f"{filename}_SCF_v3.png"
    )

    plt.savefig(
        output,
        dpi=180
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def analyze(filename):

    print(
        f"\nAnalyzing {filename}"
    )

    iq = load_signal(filename)

    frequencies, alpha_values, coherence = (
        compute_scf(iq)
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