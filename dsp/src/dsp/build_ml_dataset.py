from pathlib import Path
import os

_RF_ROOT = Path(os.environ.get("RF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))  # dsp/ klasoru
import json

import numpy as np
import pandas as pd

from .pipeline import process_iq_file


# ============================================================
# PATHS
# ============================================================

IQ_DIR = Path(os.environ.get("RF_IQ_DIR", "MATLAB_Dataset/IQ"))
META_DIR = Path(os.environ.get("RF_META_DIR", "MATLAB_Dataset/Metadata"))

OUTPUT_DIR = (_RF_ROOT / "outputs" / "datasets" / "ml_handoff")

SPECTROGRAM_DIR = OUTPUT_DIR / "spectrograms"


# ============================================================
# DATASET PARAMETERS
# ============================================================

WINDOW_SIZE = 131072
NUM_WINDOWS = 4

NFFT = 256
HOP_LENGTH = 256

EXPECTED_SHAPE = (256, 512)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        if not value:
            return np.nan

        return float(value)

    except (ValueError, TypeError):
        return np.nan


def get_label(signal_type: str) -> int:
    signal_type = str(signal_type).strip().upper()

    if signal_type == "LTE":
        return 0

    if signal_type == "LTE_DSSS":
        return 1

    raise ValueError(
        f"Unknown Signal_Type: {signal_type}"
    )


def find_metadata_file(iq_file: Path) -> Path:
    metadata_file = META_DIR / f"{iq_file.name}.csv"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found for {iq_file.name}: "
            f"{metadata_file}"
        )

    return metadata_file


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SPECTROGRAM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    iq_files = sorted(
        [
            p for p in IQ_DIR.iterdir()
            if p.is_file()
        ]
    )

    print("=" * 60)
    print("BUILDING ML DATASET")
    print("=" * 60)

    print(f"IQ files found: {len(iq_files)}")

    records = []

    for index, iq_file in enumerate(iq_files, start=1):

        print(
            f"[{index}/{len(iq_files)}] "
            f"{iq_file.name}"
        )

        metadata_file = find_metadata_file(iq_file)

        metadata = pd.read_csv(
            metadata_file
        )

        if len(metadata) != 1:
            raise ValueError(
                f"Expected exactly one metadata row for "
                f"{iq_file.name}, got {len(metadata)}"
            )

        row = metadata.iloc[0]

        signal_type = str(
            row["Signal_Type"]
        ).strip()

        label = get_label(
            signal_type
        )

        sample_rate = safe_float(
            row["LTE_SR"]
        )

        if np.isnan(sample_rate):

            sample_rate = safe_float(
                row["DSSS_SR"]
            )

        if np.isnan(sample_rate):
            raise ValueError(
                f"Sample rate missing for "
                f"{iq_file.name}"
            )

        # ----------------------------------------------------
        # PROCESS IQ
        # ----------------------------------------------------

        spectrograms = process_iq_file(
            iq_file,
            sample_rate,
        )

        if spectrograms.shape != (
            NUM_WINDOWS,
            *EXPECTED_SHAPE,
        ):
            raise ValueError(
                f"Unexpected output shape for "
                f"{iq_file.name}: "
                f"{spectrograms.shape}"
            )

        # ----------------------------------------------------
        # SAVE WINDOWS
        # ----------------------------------------------------

        for window_index in range(NUM_WINDOWS):

            spectrogram = spectrograms[
                window_index
            ]

            output_name = (
                f"{iq_file.name}"
                f"_w{window_index}.npy"
            )

            output_file = (
                SPECTROGRAM_DIR /
                output_name
            )

            np.save(
                output_file,
                spectrogram,
            )

            records.append(
                {
                    "file": iq_file.name,
                    "window": window_index,
                    "spectrogram": (
                        f"spectrograms/"
                        f"{output_name}"
                    ),
                    "label": label,
                    "signal_type": signal_type,
                    "sample_rate": sample_rate,
                    "snr_db": safe_float(
                        row["LTE_SNR_dB"]
                    ),
                    "sir_db": safe_float(
                        row["LTE_DSSS_SIR_dB"]
                    ),
                }
            )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_df = pd.DataFrame(
        records
    )

    metadata_file = (
        OUTPUT_DIR / "metadata.csv"
    )

    metadata_df.to_csv(
        metadata_file,
        index=False,
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config = {
        "window_size": WINDOW_SIZE,
        "num_windows_per_recording": NUM_WINDOWS,
        "window_function": "hann",
        "nfft": NFFT,
        "hop_length": HOP_LENGTH,
        "overlap": 0,
        "boundary": None,
        "padded": False,
        "two_sided": True,
        "fftshift": True,
        "normalization": {
            "input_db_min": -50.0,
            "input_db_max": 0.0,
            "output_min": 0.0,
            "output_max": 1.0,
        },
        "dtype": "float32",
        "spectrogram_shape": list(
            EXPECTED_SHAPE
        ),
        "labels": {
            "0": "OnlyLTE",
            "1": "LTE+DSSS",
        },
    }

    config_file = (
        OUTPUT_DIR / "dataset_config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("DATASET COMPLETE")
    print("=" * 60)

    print(
        f"Recordings: {len(iq_files)}"
    )

    print(
        f"Spectrograms: {len(records)}"
    )

    print(
        f"OnlyLTE: "
        f"{(metadata_df['label'] == 0).sum()}"
    )

    print(
        f"LTE+DSSS: "
        f"{(metadata_df['label'] == 1).sum()}"
    )

    print(
        f"Spectrogram shape: "
        f"{EXPECTED_SHAPE}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()