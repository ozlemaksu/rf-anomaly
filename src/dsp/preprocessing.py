import numpy as np


def normalize_stft_db(
    stft_db: np.ndarray,
    min_db: float = -50.0,
    max_db: float = 0.0,
) -> np.ndarray:
    normalized = (stft_db - min_db) / (max_db - min_db)

    return np.clip(normalized, 0.0, 1.0)