import numpy as np
from scipy.signal import stft


def compute_stft(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 1024,
    noverlap: int = 512,
):
    frequencies, times, Zxx = stft(
        iq,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
    )

    frequencies = np.fft.fftshift(frequencies)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    return frequencies, times, magnitude, magnitude_db