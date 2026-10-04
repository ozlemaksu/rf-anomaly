from pathlib import Path
import os

_RF_ROOT = Path(os.environ.get("RF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))  # dsp/ klasoru
import sys

import numpy as np
import pandas as pd


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
    / "scf_v4"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

N_SAMPLES = 131072

SEGMENT_LENGTH = 4096
OVERLAP = 2048

NFFT = 4096

MAX_SHIFT = 512

SMOOTHING_BINS = 5

# At Fs=30.72 MHz and NFFT=4096:
#
# df = 7.5 kHz
# alpha resolution = 2*df = 15 kHz
#
# 1.2 MHz / 15 kHz = 80 bins

TARGET_ALPHA_HZ = 1_200_000

MIN_SPACING_HZ = 150_000
MAX_SPACING_HZ = 3_000_000


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value):

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
            f"Empty metadata: {csv_path}"
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
            f"Unknown signal type: {stem}"
        )

    # --------------------------------------------------------
    # SAMPLE RATE
    # --------------------------------------------------------

    fs = np.nan

    if "LTE_SR" in df.columns:

        fs = safe_float(
            row["LTE_SR"]
        )

    # Fallback
    if np.isnan(fs) and "DSSS_SR" in df.columns:

        fs = safe_float(
            row["DSSS_SR"]
        )

    if np.isnan(fs):

        raise ValueError(
            f"Sampling rate unavailable: {stem}"
        )

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

    return fs, label, snr, sir


# ============================================================
# LOAD IQ
# ============================================================

def load_signal(stem):

    iq_path = IQ_DIR / stem

    iq = load_iq(
        iq_path
    )

    if len(iq) < N_SAMPLES:

        raise ValueError(
            f"{stem}: signal too short"
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
# COMPUTE SCF ALPHA PROFILE
# ============================================================

def compute_alpha_profile(
    iq,
    fs
):

    segments = create_segments(
        iq
    )

    # FFT
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

    alpha_step = (
        2
        * df
    )

    alpha_values = (
        np.arange(
            1,
            MAX_SHIFT + 1
        )
        * alpha_step
    )

    alpha_profile = []

    # --------------------------------------------------------
    # Alpha profile
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

        S_alpha = np.mean(
            X_plus
            * np.conj(
                X_minus
            ),
            axis=0
        )

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

        # Smooth frequency direction
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

        # Robust measure of alpha ridge
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
        alpha_profile,
        alpha_step
    )


# ============================================================
# COMB SCORE
# ============================================================

def calculate_comb_score(
    alpha_values,
    alpha_profile,
    target_spacing_hz
):

    if len(alpha_profile) < 10:

        return np.nan, 0

    alpha_step = (
        alpha_values[1]
        - alpha_values[0]
    )

    target_bins = round(
        target_spacing_hz
        / alpha_step
    )

    if target_bins <= 0:

        return np.nan, 0

    # --------------------------------------------------------
    # Normalize profile
    # --------------------------------------------------------

    median = np.median(
        alpha_profile
    )

    mad = np.median(
        np.abs(
            alpha_profile
            - median
        )
    )

    scale = (
        1.4826 * mad
        + 1e-12
    )

    normalized = (
        alpha_profile
        - median
    ) / scale

    # --------------------------------------------------------
    # Compare alpha positions separated by target spacing
    #
    # Example:
    #
    # 1.2 MHz
    # 2.4 MHz
    # 3.6 MHz
    # ...
    # --------------------------------------------------------

    positions = []

    current = (
        target_bins - 1
    )

    while current < len(
        normalized
    ):

        positions.append(
            current
        )

        current += target_bins

    if len(positions) < 2:

        return np.nan, 0

    comb_values = (
        normalized[
            positions
        ]
    )

    # Mean strength of expected comb lines
    comb_score = float(
        np.mean(
            comb_values
        )
    )

    return (
        comb_score,
        len(positions)
    )


# ============================================================
# FIND BEST SPACING
# ============================================================

def find_best_spacing(
    alpha_values,
    alpha_profile
):

    if len(alpha_profile) < 20:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    alpha_step = (
        alpha_values[1]
        - alpha_values[0]
    )

    min_bin = max(
        1,
        round(
            MIN_SPACING_HZ
            / alpha_step
        )
    )

    max_bin = min(
        len(alpha_profile) // 4,
        round(
            MAX_SPACING_HZ
            / alpha_step
        )
    )

    # Robust normalization
    median = np.median(
        alpha_profile
    )

    mad = np.median(
        np.abs(
            alpha_profile
            - median
        )
    )

    scale = (
        1.4826 * mad
        + 1e-12
    )

    profile = (
        alpha_profile
        - median
    ) / scale

    best_score = -np.inf
    best_spacing_bin = None

    # --------------------------------------------------------
    # Test possible periodic spacings
    # --------------------------------------------------------

    for spacing_bin in range(
        min_bin,
        max_bin + 1
    ):

        # Need enough repetitions
        positions = np.arange(
            spacing_bin - 1,
            len(profile),
            spacing_bin
        )

        if len(positions) < 3:
            continue

        score = np.mean(
            profile[
                positions
            ]
        )

        if score > best_score:

            best_score = score

            best_spacing_bin = (
                spacing_bin
            )

    if best_spacing_bin is None:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    best_spacing_hz = (
        best_spacing_bin
        * alpha_step
    )

    return (
        float(best_spacing_hz),
        float(best_score),
        int(best_spacing_bin)
    )


# ============================================================
# TARGET ALPHA ENERGY
# ============================================================

def calculate_target_energy(
    alpha_values,
    alpha_profile
):

    alpha_step = (
        alpha_values[1]
        - alpha_values[0]
    )

    target_bin = round(
        TARGET_ALPHA_HZ
        / alpha_step
    )

    # 1.2, 2.4, 3.6, ...
    positions = np.arange(
        target_bin - 1,
        len(alpha_profile),
        target_bin
    )

    if len(positions) == 0:

        return np.nan

    target_values = (
        alpha_profile[
            positions
        ]
    )

    baseline = np.median(
        alpha_profile
    )

    if baseline <= 0:

        return np.nan

    return float(
        np.mean(
            target_values
        )
        /
        baseline
    )


# ============================================================
# ANALYZE ONE FILE
# ============================================================

def analyze_file(
    stem,
    fs,
    label,
    snr,
    sir
):

    iq = load_signal(
        stem
    )

    (
        alpha_values,
        alpha_profile,
        alpha_step
    ) = compute_alpha_profile(
        iq,
        fs
    )

    # --------------------------------------------------------
    # Target 1.2 MHz comb
    # --------------------------------------------------------

    comb_score, comb_count = (
        calculate_comb_score(
            alpha_values,
            alpha_profile,
            TARGET_ALPHA_HZ
        )
    )

    # --------------------------------------------------------
    # Best periodic spacing
    # --------------------------------------------------------

    (
        best_spacing_hz,
        best_spacing_score,
        best_spacing_bin
    ) = find_best_spacing(
        alpha_values,
        alpha_profile
    )

    # --------------------------------------------------------
    # Target energy ratio
    # --------------------------------------------------------

    target_energy_ratio = (
        calculate_target_energy(
            alpha_values,
            alpha_profile
        )
    )

    # --------------------------------------------------------
    # General alpha statistics
    # --------------------------------------------------------

    baseline = np.median(
        alpha_profile
    )

    maximum = np.max(
        alpha_profile
    )

    mean = np.mean(
        alpha_profile
    )

    # --------------------------------------------------------
    # Peak location
    # --------------------------------------------------------

    max_index = np.argmax(
        alpha_profile
    )

    max_alpha_hz = (
        alpha_values[
            max_index
        ]
    )

    return {

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

        "alpha_step_hz":
            alpha_step,

        "max_alpha_hz":
            float(
                max_alpha_hz
            ),

        "alpha_profile_mean":
            float(
                mean
            ),

        "alpha_profile_median":
            float(
                baseline
            ),

        "alpha_profile_max":
            float(
                maximum
            ),

        "max_to_median":
            float(
                maximum
                /
                (baseline + 1e-12)
            ),

        "target_1p2mhz_comb_score":
            comb_score,

        "target_1p2mhz_count":
            comb_count,

        "target_energy_ratio":
            target_energy_ratio,

        "best_spacing_hz":
            best_spacing_hz,

        "best_spacing_mhz":
            (
                best_spacing_hz / 1e6
                if not np.isnan(
                    best_spacing_hz
                )
                else np.nan
            ),

        "best_spacing_score":
            best_spacing_score,

        "best_spacing_bin":
            best_spacing_bin
    }


# ============================================================
# COLLECT FILES
# ============================================================

def collect_files():

    files = []

    print(
        "\n===== COLLECTING 30.72 MHz FILES ====="
    )

    for iq_file in sorted(
        IQ_DIR.iterdir()
    ):

        if not iq_file.is_file():
            continue

        stem = iq_file.name

        if not (
            stem.startswith("OnlyLTE")
            or
            stem.startswith(
                "Combined_LTE_DSSS"
            )
        ):

            continue

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
                f"SKIP: {stem} -> {e}"
            )

            continue

        if np.isclose(
            fs,
            30_720_000,
            rtol=0,
            atol=1
        ):

            files.append(
                (
                    stem,
                    fs,
                    label,
                    snr,
                    sir
                )
            )

    normal = [
        x for x in files
        if x[2] == 0
    ]

    anomaly = [
        x for x in files
        if x[2] == 1
    ]

    print(
        f"OnlyLTE: {len(normal)}"
    )

    print(
        f"LTE+DSSS: {len(anomaly)}"
    )

    print(
        f"Total: {len(files)}"
    )

    return files


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df):

    print(
        "\n\n=========================================="
    )

    print(
        "              GROUP SUMMARY"
    )

    print(
        "=========================================="
    )

    summary_columns = [

        "target_1p2mhz_comb_score",
        "target_energy_ratio",
        "best_spacing_mhz",
        "best_spacing_score",
        "max_to_median"
    ]

    summary = (
        df
        .groupby(
            "label"
        )[
            summary_columns
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std"
            ]
        )
    )

    print(
        summary.to_string()
    )

    # --------------------------------------------------------
    # SIR groups for anomaly class
    # --------------------------------------------------------

    anomaly = df[
        df["label"] == 1
    ].copy()

    if not anomaly.empty:

        print(
            "\n\n=========================================="
        )

        print(
            "           LTE+DSSS BY SIR"
        )

        print(
            "=========================================="
        )

        sir_summary = (
            anomaly
            .groupby(
                "sir_db"
            )[
                summary_columns
            ]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "std"
                ]
            )
        )

        print(
            sir_summary.to_string()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    files = collect_files()

    if len(files) == 0:

        print(
            "\nNo 30.72 MHz files found."
        )

        return

    results = []

    total = len(files)

    print(
        "\n===== STARTING SCF V4 ====="
    )

    print(
        f"Files to analyze: {total}"
    )

    for index, item in enumerate(
        files,
        start=1
    ):

        (
            stem,
            fs,
            label,
            snr,
            sir
        ) = item

        print(
            f"\n[{index}/{total}] "
            f"{stem}"
        )

        try:

            result = analyze_file(
                stem,
                fs,
                label,
                snr,
                sir
            )

            results.append(
                result
            )

            print(
                f"  label = {label}"
            )

            print(
                f"  SNR = {snr}"
            )

            if np.isnan(sir):

                print(
                    "  SIR = NaN"
                )

            else:

                print(
                    f"  SIR = {sir}"
                )

            print(
                "  1.2 MHz comb score = "
                f"{result['target_1p2mhz_comb_score']:.4f}"
            )

            print(
                "  target energy ratio = "
                f"{result['target_energy_ratio']:.4f}"
            )

            print(
                "  best spacing = "
                f"{result['best_spacing_mhz']:.3f} MHz"
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # SAVE ALL RESULTS
    # ========================================================

    output_csv = (
        OUT_DIR
        / "scf_features_v4_all_30MHz.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        "\n=========================================="
    )

    print(
        f"Saved: {output_csv}"
    )

    print(
        "=========================================="
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    if not df.empty:

        print_summary(
            df
        )

    # ========================================================
    # SAVE COMPACT SUMMARY
    # ========================================================

    compact_columns = [

        "file",
        "label",
        "snr_db",
        "sir_db",

        "target_1p2mhz_comb_score",

        "target_energy_ratio",

        "best_spacing_mhz",

        "best_spacing_score",

        "max_to_median"
    ]

    compact_path = (
        OUT_DIR
        / "scf_features_v4_compact.csv"
    )

    df[
        compact_columns
    ].to_csv(
        compact_path,
        index=False
    )

    print(
        f"\nCompact results saved:"
    )

    print(
        compact_path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()