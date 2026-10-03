import numpy as np

from .io import load_iq
from .stft import compute_stft
from .preprocessing import normalize_stft_db


def process_iq_file(
    iq_file,
    sample_rate: float,
    nperseg: int = 1024,
    noverlap: int = 512,
) -> np.ndarray:
    iq = load_iq(iq_file)

    _, _, _, stft_db = compute_stft(
        iq,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    spectrogram = normalize_stft_db(stft_db)

    spectrogram = np.fft.fftshift(
        spectrogram,
        axes=0,
    )

    return spectrogram.astype(np.float32)