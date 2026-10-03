from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\Users\asus\Desktop\RF-Anomaly-Detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.io import load_iq


IQ_DIR = Path(r"C:\Users\asus\Desktop\MATLAB_Dataset\IQ")
OUT_DIR = PROJECT_ROOT / "outputs" / "scf_analysis"

FS = 7_680_000

# AI pipeline ile aynı pencere uzunluğu
N_SAMPLES = 131072

# Segment FFT
NFFT = 2048

# Segment uzunluğu
SEGMENT_LENGTH = 2048

# Segmentler arası overlap
OVERLAP = 1024

# İncelenecek cyclic frequency sayısı
N_ALPHA = 64


def prepare_signal(filename):
    """IQ dosyasını yükle ve analiz penceresini hazırla."""

    iq = load_iq(IQ_DIR / filename)

    iq = iq[:N_SAMPLES]

    # DC kaldır
    iq = iq - np.mean(iq)

    # Güç normalize
    power = np.mean(np.abs(iq) ** 2)
    iq = iq / np.sqrt(power + 1e-12)

    return iq


def compute_scf(iq):
    """
    Basitleştirilmiş Spectral Correlation Function hesabı.

    S_x^alpha(f) ≈ E[
        X(f + alpha/2) *
        X*(f - alpha/2)
    ]
    """

    step = SEGMENT_LENGTH - OVERLAP

    n_segments = (
        (len(iq) - SEGMENT_LENGTH) // step
    ) + 1

    window = np.hanning(SEGMENT_LENGTH)

    # Segment FFT'lerini tut
    spectra = []

    for i in range(n_segments):

        start = i * step

        segment = iq[
            start:start + SEGMENT_LENGTH
        ]

        segment = segment * window

        X = np.fft.fftshift(
            np.fft.fft(segment, n=NFFT)
        )

        spectra.append(X)

    spectra = np.asarray(spectra)

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            NFFT,
            d=1 / FS
        )
    )

    # Cyclic frequency ekseni
    max_alpha = FS / 4

    alpha_values = np.linspace(
        -max_alpha,
        max_alpha,
        N_ALPHA
    )

    scf = np.zeros(
        (N_ALPHA, NFFT),
        dtype=np.float64
    )

    for a_idx, alpha in enumerate(alpha_values):

        # alpha / 2'nin frekans bin karşılığı
        shift = int(
            round(
                (alpha / 2)
                / (FS / NFFT)
            )
        )

        center = NFFT // 2

        f_plus_start = center + shift
        f_minus_start = center - shift

        if (
            f_plus_start < 0
            or f_plus_start >= NFFT
            or f_minus_start < 0
            or f_minus_start >= NFFT
        ):
            continue

        # Frekans indekslerini eşleştir
        valid = min(
            NFFT - abs(shift),
            NFFT
        )

        if shift >= 0:

            x_plus = spectra[
                :,
                shift:
            ]

            x_minus = spectra[
                :,
                :-shift if shift != 0 else None
            ]

        else:

            s = abs(shift)

            x_plus = spectra[
                :,
                :-s
            ]

            x_minus = spectra[
                :,
                s:
            ]

        length = min(
            x_plus.shape[1],
            x_minus.shape[1]
        )

        if length <= 0:
            continue

        correlation = np.mean(
            x_plus[:, :length]
            *
            np.conj(
                x_minus[:, :length]
            ),
            axis=0
        )

        # Merkeze yerleştir
        if shift >= 0:
            start_idx = center
            end_idx = min(
                center + length,
                NFFT
            )

            scf[
                a_idx,
                start_idx:end_idx
            ] = np.abs(
                correlation[:end_idx - start_idx]
            )

        else:
            start_idx = max(
                center - length,
                0
            )
            end_idx = center

            scf[
                a_idx,
                start_idx:end_idx
            ] = np.abs(
                correlation[-(end_idx - start_idx):]
            )

    return frequencies, alpha_values, scf


def save_scf_plot(
    filename,
    frequencies,
    alpha_values,
    scf
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # dB dönüşümü
    scf_db = 20 * np.log10(
        scf + 1e-12
    )

    # Normalize
    scf_db -= np.max(scf_db)

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        scf_db,
        aspect="auto",
        origin="lower",
        extent=[
            frequencies[0] / 1e6,
            frequencies[-1] / 1e6,
            alpha_values[0] / 1e6,
            alpha_values[-1] / 1e6
        ],
        cmap="viridis"
    )

    plt.colorbar(
        label="Normalized SCF magnitude (dB)"
    )

    plt.xlabel(
        "Spectral frequency f (MHz)"
    )

    plt.ylabel(
        "Cyclic frequency α (MHz)"
    )

    plt.title(
        f"Spectral Correlation Function - {filename}"
    )

    plt.tight_layout()

    output = (
        OUT_DIR /
        f"{filename}_SCF.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(f"Saved: {output}")


def analyze_file(filename):

    print(f"\nAnalyzing: {filename}")

    iq = prepare_signal(filename)

    frequencies, alpha_values, scf = compute_scf(
        iq
    )

    save_scf_plot(
        filename,
        frequencies,
        alpha_values,
        scf
    )


if __name__ == "__main__":

    analyze_file(
        "OnlyLTE_frame_100"
    )

    analyze_file(
        "Combined_LTE_DSSS_frame_264"
    )