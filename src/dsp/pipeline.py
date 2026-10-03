from pathlib import Path

import numpy as np

from .io import load_iq
from .stft import compute_stft
from .preprocessing import normalize_stft_db


WINDOW_SIZE = 131072
NUM_WINDOWS = 4

NFFT = 256
HOP_LENGTH = 256


def get_window_starts(signal_length: int) -> np.ndarray:
    """
    Divide a recording into four evenly distributed windows.
    """

    if signal_length < WINDOW_SIZE:
        raise ValueError(
            f"Signal length ({signal_length}) is smaller than "
            f"window size ({WINDOW_SIZE})."
        )

    return np.linspace(
        0,
        signal_length - WINDOW_SIZE,
        NUM_WINDOWS,
        dtype=int,
    )


def process_window(
    iq_window: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """
    Convert one IQ window into an AI-ready spectrogram.

    Output shape:
        (256, 512)
    """

    _, _, _, stft_db = compute_stft(
        iq_window,
        sample_rate,
        nfft=NFFT,
        hop_length=HOP_LENGTH,
    )

    spectrogram = normalize_stft_db(stft_db)

    spectrogram = spectrogram.astype(np.float32)

    expected_shape = (256, 512)

    if spectrogram.shape != expected_shape:
        raise ValueError(
            f"Unexpected spectrogram shape: {spectrogram.shape}. "
            f"Expected {expected_shape}."
        )

    return spectrogram


def process_iq_file(
    iq_file: str | Path,
    sample_rate: float,
) -> np.ndarray:
    """
    Process one complete IQ recording.

    Returns:
        np.ndarray with shape (4, 256, 512)

    Dimensions:
        4   = windows
        256 = frequency bins
        512 = time bins
    """

    iq = load_iq(iq_file)

    starts = get_window_starts(len(iq))

    spectrograms = []

    for start in starts:
        iq_window = iq[start:start + WINDOW_SIZE]

        spectrogram = process_window(
            iq_window,
            sample_rate,
        )

        spectrograms.append(spectrogram)

    return np.stack(spectrograms).astype(np.float32)