import numpy as np
from scipy.signal import stft


def compute_stft(
    iq: np.ndarray,
    sample_rate: float,
    nfft: int = 256,
    hop_length: int = 256,
):
    """
    Compute a two-sided, fftshifted STFT.

    AI pipeline parameters:
        nfft = 256
        hop_length = 256
        Hann window
        zero overlap
    """

    nperseg = nfft
    noverlap = nperseg - hop_length

    frequencies, times, Zxx = stft(
        iq,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    frequencies = np.fft.fftshift(frequencies)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    return frequencies, times, magnitude, magnitude_db