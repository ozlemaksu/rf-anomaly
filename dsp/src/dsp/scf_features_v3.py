from pathlib import Path
import os

_RF_ROOT = Path(os.environ.get("RF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))  # dsp/ klasoru
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = _RF_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.io import load_iq


IQ_DIR = Path(os.environ.get("RF_IQ_DIR", "MATLAB_Dataset/IQ"))

META_DIR = Path(os.environ.get("RF_META_DIR", "MATLAB_Dataset/Metadata"))

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "scf_v3"
    / "features"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SCF PARAMETERS
# ============================================================

N_SAMPLES = 131072

SEGMENT_LENGTH = 4096
OVERLAP = 2048

NFFT = 4096

MAX_SHIFT = 512

SMOOTHING_BINS = 5


# ============================================================
# HELPER: SAFE NUMERIC CONVERSION
# ============================================================

def safe_float(value):
    """
    Convert metadata value to float safely.

    Handles:
    - normal numbers
    - strings
    - whitespace
    - empty strings
    - NaN
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    try:
        return float(text)
    except ValueError:
        return np.nan


# ============================================================
# METADATA
# ============================================================

def get_metadata(stem):

    csv_path = META_DIR / f"{stem}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        skipinitialspace=True
    )

    if df.empty:
        raise ValueError(
            f"Empty metadata file: {csv_path}"
        )

    row = df.iloc[0]

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    if stem.startswith("OnlyLTE"):

        label = 0

    elif stem.startswith("Combined_LTE_DSSS"):

        label = 1

    else:

        raise ValueError(
            f"Unknown file type: {stem}"
        )

    # --------------------------------------------------------
    # LTE SAMPLE RATE
    # --------------------------------------------------------

    fs = np.nan

    if "LTE_SR" in df.columns:

        fs = safe_float(
            row["LTE_SR"]
        )

    # --------------------------------------------------------
    # DSSS SAMPLE RATE
    # --------------------------------------------------------

    dsss_sr = np.nan

    if "DSSS_SR" in df.columns:

        dsss_sr = safe_float(
            row["DSSS_SR"]
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # For Combined files LTE_SR should contain the LTE rate.
    #
    # For OnlyLTE files, if LTE_SR is blank, try other
    # explicitly available sampling-rate fields.
    # --------------------------------------------------------

    if np.isnan(fs):

        if not np.isnan(dsss_sr):

            fs = dsss_sr

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    snr = np.nan

    if "LTE_SNR_dB" in df.columns:

        snr = safe_float(
            row["LTE_SNR_dB"]
        )

    # --------------------------------------------------------
    # SIR
    # --------------------------------------------------------

    sir = np.nan

    if "LTE_DSSS_SIR_dB" in df.columns:

        sir = safe_float(
            row["LTE_DSSS_SIR_dB"]
        )

    # --------------------------------------------------------
    # If sampling rate still unavailable,
    # print all metadata columns for diagnosis.
    # --------------------------------------------------------

    if np.isnan(fs):

        print(
            "\nWARNING: Sampling rate not found:"
        )

        print(
            f"File: {stem}"
        )

        print(
            "Available metadata:"
        )

        for column in df.columns:

            print(
                f"  {column}: "
                f"{repr(row[column])}"
            )

        raise ValueError(
            f"Could not determine sampling rate: {stem}"
        )

    return (
        fs,
        label,
        snr,
        sir
    )


# ============================================================
# LOAD SIGNAL
# ============================================================

def load_signal(stem):

    iq_path = IQ_DIR / stem

    if not iq_path.exists():

        raise FileNotFoundError(
            f"IQ file not found: {iq_path}"
        )

    iq = load_iq(
        iq_path
    )

    if len(iq) < N_SAMPLES:

        raise ValueError(
            f"{stem}: signal shorter than "
            f"{N_SAMPLES} samples"
        )

    iq = iq[
        :N_SAMPLES
    ]

    # Remove DC
    iq = (
        iq
        - np.mean(iq)
    )

    # RMS normalization
    rms = np.sqrt(
        np.mean(
            np.abs(iq) ** 2
        )
    )

    iq = (
        iq
        / (rms + 1e-12)
    )

    return iq


# ============================================================
# CREATE SEGMENTS
# ============================================================

def create_segments(iq):

    step = (
        SEGMENT_LENGTH
        - OVERLAP
    )

    if len(iq) < SEGMENT_LENGTH:

        raise ValueError(
            "Signal shorter than SEGMENT_LENGTH"
        )

    n_segments = (
        (
            len(iq)
            - SEGMENT_LENGTH
        )
        // step
    ) + 1

    window = np.hanning(
        SEGMENT_LENGTH
    )

    segments = []

    for i in range(
        n_segments
    ):

        start = i * step

        segment = iq[
            start:
            start + SEGMENT_LENGTH
        ]

        segment = (
            segment
            * window
        )

        segments.append(
            segment
        )

    return np.asarray(
        segments
    )


# ============================================================
# COMPUTE ALPHA PROFILE
# ============================================================

def compute_alpha_profile(
    iq,
    fs
):

    segments = create_segments(
        iq
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    X = np.fft.fft(
        segments,
        n=NFFT,
        axis=1
    )

    X = np.fft.fftshift(
        X,
        axes=1
    )

    df = (
        fs
        / NFFT
    )

    # --------------------------------------------------------
    # Alpha grid
    #
    # alpha = 2 * m * df
    # --------------------------------------------------------

    alpha_values = (
        2
        * np.arange(
            1,
            MAX_SHIFT + 1
        )
        * df
    )

    alpha_profile = []

    # --------------------------------------------------------
    # SCF / spectral coherence
    # --------------------------------------------------------

    for m in range(
        1,
        MAX_SHIFT + 1
    ):

        length = (
            NFFT
            - 2 * m
        )

        if length <= 0:
            break

        X_plus = X[
            :,
            2 * m:
        ]

        X_minus = X[
            :,
            :length
        ]

        # ----------------------------------------------------
        # Cross spectral term
        # ----------------------------------------------------

        S_alpha = np.mean(
            X_plus
            * np.conj(
                X_minus
            ),
            axis=0
        )

        # ----------------------------------------------------
        # Power
        # ----------------------------------------------------

        P_plus = np.mean(
            np.abs(
                X_plus
            ) ** 2,
            axis=0
        )

        P_minus = np.mean(
            np.abs(
                X_minus
            ) ** 2,
            axis=0
        )

        # ----------------------------------------------------
        # Spectral coherence
        # ----------------------------------------------------

        coherence = (
            np.abs(
                S_alpha
            ) ** 2
            /
            (
                P_plus
                * P_minus
                + 1e-12
            )
        )

        coherence = np.clip(
            coherence,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Smoothing
        # ----------------------------------------------------

        if SMOOTHING_BINS > 1:

            kernel = (
                np.ones(
                    SMOOTHING_BINS
                )
                /
                SMOOTHING_BINS
            )

            coherence = np.convolve(
                coherence,
                kernel,
                mode="same"
            )

        # ----------------------------------------------------
        # High percentile
        # ----------------------------------------------------

        profile_value = np.percentile(
            coherence,
            95
        )

        alpha_profile.append(
            profile_value
        )

    alpha_values = alpha_values[
        :len(alpha_profile)
    ]

    alpha_profile = np.asarray(
        alpha_profile
    )

    return (
        alpha_values,
        alpha_profile
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(
    alpha_values,
    alpha_profile
):

    if len(alpha_profile) == 0:

        return {

            "alpha_peak_hz":
                np.nan,

            "alpha_peak_coherence":
                np.nan,

            "alpha_peak_count":
                0,

            "alpha_spacing_hz":
                np.nan,

            "alpha_spacing_std_hz":
                np.nan,

            "nonzero_alpha_mean":
                np.nan,

            "nonzero_alpha_max":
                np.nan,

            "active_alpha_count":
                0
        }

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = np.median(
        alpha_profile
    )

    noise_std = np.std(
        alpha_profile
    )

    prominence = max(
        noise_std * 1.5,
        0.01
    )

    # --------------------------------------------------------
    # Peaks
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        alpha_profile,
        prominence=prominence,
        distance=3
    )

    # --------------------------------------------------------
    # No peaks
    # --------------------------------------------------------

    if len(peaks) == 0:

        active_threshold = (
            baseline
            + 2 * noise_std
        )

        active_count = int(
            np.sum(
                alpha_profile
                > active_threshold
            )
        )

        return {

            "alpha_peak_hz":
                np.nan,

            "alpha_peak_coherence":
                np.nan,

            "alpha_peak_count":
                0,

            "alpha_spacing_hz":
                np.nan,

            "alpha_spacing_std_hz":
                np.nan,

            "nonzero_alpha_mean":
                float(
                    np.mean(
                        alpha_profile
                    )
                ),

            "nonzero_alpha_max":
                float(
                    np.max(
                        alpha_profile
                    )
                ),

            "active_alpha_count":
                active_count
        }

    # --------------------------------------------------------
    # Strongest peak
    # --------------------------------------------------------

    peak_values = (
        alpha_profile[
            peaks
        ]
    )

    strongest_index = peaks[
        np.argmax(
            peak_values
        )
    ]

    strongest_value = (
        alpha_profile[
            strongest_index
        ]
    )

    strongest_alpha = (
        alpha_values[
            strongest_index
        ]
    )

    # --------------------------------------------------------
    # Peak spacing
    # --------------------------------------------------------

    peak_alphas = (
        alpha_values[
            peaks
        ]
    )

    spacing = np.diff(
        peak_alphas
    )

    if len(spacing) > 0:

        spacing_median = float(
            np.median(
                spacing
            )
        )

        spacing_std = float(
            np.std(
                spacing
            )
        )

    else:

        spacing_median = np.nan
        spacing_std = np.nan

    # --------------------------------------------------------
    # Active alpha count
    # --------------------------------------------------------

    active_threshold = (
        baseline
        + 2 * noise_std
    )

    active_count = int(
        np.sum(
            alpha_profile
            > active_threshold
        )
    )

    return {

        "alpha_peak_hz":
            float(
                strongest_alpha
            ),

        "alpha_peak_coherence":
            float(
                strongest_value
            ),

        "alpha_peak_count":
            int(
                len(peaks)
            ),

        "alpha_spacing_hz":
            spacing_median,

        "alpha_spacing_std_hz":
            spacing_std,

        "nonzero_alpha_mean":
            float(
                np.mean(
                    alpha_profile
                )
            ),

        "nonzero_alpha_max":
            float(
                np.max(
                    alpha_profile
                )
            ),

        "active_alpha_count":
            active_count
    }


# ============================================================
# SELECT FILES
# ============================================================

def select_files():

    normal = []
    anomaly = []

    print(
        "\n===== SEARCHING FILES ====="
    )

    for iq_file in sorted(
        IQ_DIR.iterdir()
    ):

        if not iq_file.is_file():
            continue

        stem = iq_file.name

        # ----------------------------------------------------
        # OnlyLTE
        # ----------------------------------------------------

        if stem.startswith(
            "OnlyLTE"
        ):

            try:

                (
                    fs,
                    label,
                    snr,
                    sir
                ) = get_metadata(
                    stem
                )

            except Exception as e:

                print(
                    f"Metadata error: "
                    f"{stem} -> {e}"
                )

                continue

            if np.isclose(
                fs,
                30_720_000,
                rtol=0,
                atol=1
            ):

                normal.append(
                    (
                        stem,
                        fs,
                        label,
                        snr,
                        sir
                    )
                )

        # ----------------------------------------------------
        # LTE + DSSS
        # ----------------------------------------------------

        elif stem.startswith(
            "Combined_LTE_DSSS"
        ):

            try:

                (
                    fs,
                    label,
                    snr,
                    sir
                ) = get_metadata(
                    stem
                )

            except Exception as e:

                print(
                    f"Metadata error: "
                    f"{stem} -> {e}"
                )

                continue

            if np.isclose(
                fs,
                30_720_000,
                rtol=0,
                atol=1
            ):

                anomaly.append(
                    (
                        stem,
                        fs,
                        label,
                        snr,
                        sir
                    )
                )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    print(
        f"\n30.72 MHz OnlyLTE count: "
        f"{len(normal)}"
    )

    print(
        f"30.72 MHz LTE+DSSS count: "
        f"{len(anomaly)}"
    )

    # --------------------------------------------------------
    # Select 6 + 6
    # --------------------------------------------------------

    selected_normal = normal[:6]

    selected_anomaly = anomaly[:6]

    print(
        "\nSelected OnlyLTE:"
    )

    for item in selected_normal:

        print(
            f"{item[0]} | "
            f"SNR={item[3]} dB"
        )

    print(
        "\nSelected LTE+DSSS:"
    )

    for item in selected_anomaly:

        print(
            f"{item[0]} | "
            f"SNR={item[3]} dB | "
            f"SIR={item[4]} dB"
        )

    return (
        selected_normal
        + selected_anomaly
    )


# ============================================================
# MAIN
# ============================================================

def main():

    selected = select_files()

    print(
        "\n================================"
    )

    print(
        f"Total selected: "
        f"{len(selected)}"
    )

    print(
        "================================"
    )

    if len(selected) == 0:

        print(
            "\nNo suitable files found."
        )

        return

    results = []

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    for (
        stem,
        fs,
        label,
        snr,
        sir
    ) in selected:

        print(
            "\n------------------------------"
        )

        print(
            f"Analyzing: {stem}"
        )

        print(
            f"Fs: "
            f"{fs / 1e6:.2f} MHz"
        )

        print(
            f"Label: {label}"
        )

        print(
            f"SNR: {snr} dB"
        )

        if np.isnan(sir):

            print(
                "SIR: NaN"
            )

        else:

            print(
                f"SIR: {sir} dB"
            )

        # ----------------------------------------------------
        # Load IQ
        # ----------------------------------------------------

        iq = load_signal(
            stem
        )

        # ----------------------------------------------------
        # Alpha profile
        # ----------------------------------------------------

        (
            alpha_values,
            alpha_profile
        ) = compute_alpha_profile(
            iq,
            fs
        )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = extract_features(
            alpha_values,
            alpha_profile
        )

        # ----------------------------------------------------
        # Readable output
        # ----------------------------------------------------

        peak_hz = (
            features[
                "alpha_peak_hz"
            ]
        )

        spacing_hz = (
            features[
                "alpha_spacing_hz"
            ]
        )

        if np.isnan(
            peak_hz
        ):

            peak_text = "NaN"

        else:

            peak_text = (
                f"{peak_hz / 1e6:.3f} MHz"
            )

        if np.isnan(
            spacing_hz
        ):

            spacing_text = "NaN"

        else:

            spacing_text = (
                f"{spacing_hz / 1e6:.3f} MHz"
            )

        print(
            f"Alpha peak: "
            f"{peak_text}"
        )

        print(
            f"Peak count: "
            f"{features['alpha_peak_count']}"
        )

        print(
            f"Alpha spacing: "
            f"{spacing_text}"
        )

        print(
            f"Peak coherence: "
            f"{features['alpha_peak_coherence']}"
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results.append({

            "file":
                stem,

            "label":
                label,

            "fs_hz":
                fs,

            "snr_db":
                snr,

            "sir_db":
                sir,

            **features
        })

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        OUT_DIR
        / "scf_features_v3_test.csv"
    )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        "\n================================"
    )

    print(
        f"Saved: "
        f"{output_path}"
    )

    print(
        "================================"
    )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    columns = [

        "file",
        "label",
        "snr_db",
        "sir_db",

        "alpha_peak_hz",

        "alpha_peak_count",

        "alpha_spacing_hz",

        "alpha_spacing_std_hz",

        "alpha_peak_coherence"
    ]

    print(
        "\n===== RESULTS ====="
    )

    print(
        df[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()