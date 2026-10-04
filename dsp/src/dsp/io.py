from pathlib import Path
import numpy as np


def load_iq(file_path: str | Path) -> np.ndarray:
    file_path = Path(file_path)

    with open(file_path, "rb") as file:
        file.seek(8)
        raw_data = np.fromfile(file, dtype=np.float32)

    I = raw_data[0::2]
    Q = raw_data[1::2]

    return I + 1j * Q