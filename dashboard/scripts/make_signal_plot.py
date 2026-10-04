import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "preprocessed"


# Verileri yükle
X_iq = np.load(DATA / "X_iq.npy")
X_spec = np.load(DATA / "X_spec.npy")


# İlk test örneğini al
idx = 0

I = X_iq[idx, 0]
Q = X_iq[idx, 1]

spec = X_spec[idx, 0]


# ---------------------------------------------------------
# Raw IQ
# ---------------------------------------------------------

plt.figure(figsize=(10, 4))

n = min(4000, len(I))

plt.plot(I[:n], label="I")
plt.plot(Q[:n], label="Q")

plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Raw IQ Signal")
plt.legend()
plt.tight_layout()

plt.savefig(
    DATA.parent / "raw_iq_example.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# STFT
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.imshow(
    spec,
    aspect="auto",
    origin="lower"
)

plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("STFT Spectrogram")

plt.colorbar(label="Magnitude")

plt.tight_layout()

plt.savefig(
    DATA.parent / "stft_example.png",
    dpi=300
)

plt.close()


print("Oluşturuldu:")
print("data/raw_iq_example.png")
print("data/stft_example.png")