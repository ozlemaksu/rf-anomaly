from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\asus\Desktop\RF-Anomaly-Detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.io import load_iq


IQ_DIR = Path(r"C:\Users\asus\Desktop\MATLAB_Dataset\IQ")
OUT_DIR = PROJECT_ROOT / "outputs" / "scf_features"

FS = 7_680_000
N_SAMPLES = 131072

NFFT = 2048
SEGMENT_LENGTH = 2048
OVERLAP = 1024

N_ALPHA = 64


def prepare_signal(filename):

    iq = load_iq(IQ_DIR / filename)

    iq = iq[:N_SAMPLES]

    iq = iq - np.mean(iq)

    power = np.mean(np.abs(iq) ** 2)

    iq = iq / np.sqrt(power + 1e-12)

    return iq


def compute_scf(iq):

    step = SEGMENT_LENGTH - OVERLAP

    n_segments = (
        (len(iq) - SEGMENT_LENGTH) // step
    ) + 1

    window = np.hanning(SEGMENT_LENGTH)

    spectra = []

    for i in range(n_segments):

        start = i * step

        segment = iq[
            start:start + SEGMENT_LENGTH
        ]

        segment = segment * window

        X = np.fft.fftshift(
            np.fft.fft(
                segment,
                n=NFFT
            )
        )

        spectra.append(X)

    spectra = np.asarray(spectra)

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            NFFT,
            d=1 / FS
        )
    )

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

    center = NFFT // 2

    for a_idx, alpha in enumerate(alpha_values):

        shift = int(
            round(
                (alpha / 2)
                / (FS / NFFT)
            )
        )

        if shift == 0:

            correlation = np.mean(
                np.abs(spectra) ** 2,
                axis=0
            )

            scf[a_idx, :] = np.abs(
                correlation
            )

            continue

        if shift > 0:

            x_plus = spectra[:, shift:]
            x_minus = spectra[:, :-shift]

        else:

            s = abs(shift)

            x_plus = spectra[:, :-s]
            x_minus = spectra[:, s:]

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

        magnitude = np.abs(
            correlation
        )

        if shift > 0:

            start = center
            end = min(
                center + length,
                NFFT
            )

            scf[
                a_idx,
                start:end
            ] = magnitude[
                :end - start
            ]

        else:

            start = max(
                center - length,
                0
            )

            end = center

            scf[
                a_idx,
                start:end
            ] = magnitude[
                -(end - start):
            ]

    return frequencies, alpha_values, scf


def extract_features(
    frequencies,
    alpha_values,
    scf
):

    scf_power = scf ** 2

    # α = 0 satırını ayır
    alpha_zero_idx = np.argmin(
        np.abs(alpha_values)
    )

    alpha_zero = scf_power[
        alpha_zero_idx
    ]

    # α != 0
    nonzero_alpha = np.delete(
        scf_power,
        alpha_zero_idx,
        axis=0
    )

    # Toplam enerji
    total_energy = np.sum(
        scf_power
    )

    nonzero_energy = np.sum(
        nonzero_alpha
    )

    # α != 0 / toplam enerji
    nonzero_ratio = (
        nonzero_energy
        /
        (total_energy + 1e-12)
    )

    # α = 0 / toplam enerji
    alpha_zero_ratio = (
        np.sum(alpha_zero)
        /
        (total_energy + 1e-12)
    )

    # En güçlü α != 0 noktası
    temp = nonzero_alpha.copy()

    max_idx = np.unravel_index(
        np.argmax(temp),
        temp.shape
    )

    max_nonzero_scf = (
        temp[max_idx]
    )

    # Global maximum
    max_scf = np.max(
        scf_power
    )

    # Normalize edilmiş max nonzero SCF
    max_nonzero_ratio = (
        max_nonzero_scf
        /
        (max_scf + 1e-12)
    )

    # α = 0 dışındaki aktif alan oranı
    threshold = (
        0.25
        *
        max_scf
    )

    active_ratio = np.mean(
        nonzero_alpha > threshold
    )

    # α eksenindeki enerji
    alpha_energy = np.sum(
        nonzero_alpha,
        axis=1
    )

    max_alpha_idx = np.argmax(
        alpha_energy
    )

    # α değerini bul
    nonzero_alpha_values = np.delete(
        alpha_values,
        alpha_zero_idx
    )

    max_alpha = (
        nonzero_alpha_values[
            max_alpha_idx
        ]
    )

    return {
        "total_scf_energy": total_energy,
        "nonzero_alpha_energy": nonzero_energy,
        "nonzero_alpha_ratio": nonzero_ratio,
        "alpha_zero_ratio": alpha_zero_ratio,
        "max_nonzero_scf": max_nonzero_scf,
        "max_nonzero_scf_ratio": max_nonzero_ratio,
        "nonzero_active_area_ratio": active_ratio,
        "max_energy_alpha_hz": max_alpha,
    }


def analyze_file(filename, label):

    print(f"Analyzing {filename}")

    iq = prepare_signal(filename)

    frequencies, alpha_values, scf = compute_scf(
        iq
    )

    features = extract_features(
        frequencies,
        alpha_values,
        scf
    )

    features["file"] = filename
    features["label"] = label

    return features


if __name__ == "__main__":

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    results.append(
        analyze_file(
            "OnlyLTE_frame_100",
            0
        )
    )

    results.append(
        analyze_file(
            "Combined_LTE_DSSS_frame_264",
            1
        )
    )

    df = pd.DataFrame(
        results
    )

    output = (
        OUT_DIR /
        "scf_features.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print()
    print(df.to_string(index=False))
    print()
    print(f"Saved: {output}")