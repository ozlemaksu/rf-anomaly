import numpy as np


def compute_fft(
    iq: np.ndarray,
    sample_rate: float,
    n_fft: int = 4096,
):
    iq_segment = iq[:n_fft]

    fft_result = np.fft.fft(iq_segment)
    fft_shifted = np.fft.fftshift(fft_result)

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(n_fft, d=1 / sample_rate)
    )

    magnitude = np.abs(fft_shifted)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    return frequencies, magnitude, magnitude_db