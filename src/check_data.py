import os
import numpy as np

DATA = "data/preprocessed"

print("=== PREPROCESSING KONTROL ===")

files = [
    "X_iq.npy",
    "X_spec.npy",
    "file_id.npy",
    "y.npy",
    "sir_db.npy",
    "fs_hz.npy",
    "file_index.csv"
]

for file in files:
    path = os.path.join(DATA, file)

    if os.path.exists(path):
        print("[OK]", file)
    else:
        print("[HATA]", file, "bulunamadi")

print("\n=== ARRAY BILGILERI ===")

X_iq = np.load(os.path.join(DATA, "X_iq.npy"), mmap_mode="r")
X_spec = np.load(os.path.join(DATA, "X_spec.npy"), mmap_mode="r")
file_id = np.load(os.path.join(DATA, "file_id.npy"))
y = np.load(os.path.join(DATA, "y.npy"))
sir = np.load(os.path.join(DATA, "sir_db.npy"))

print("X_iq:", X_iq.shape, X_iq.dtype)
print("X_spec:", X_spec.shape, X_spec.dtype)
print("file_id:", file_id.shape, file_id.dtype)
print("y:", y.shape, y.dtype)
print("sir_db:", sir.shape, sir.dtype)

print("\n=== ETIKETLER ===")
print("LTE:", int((y == 0).sum()))
print("LTE+DSSS:", int((y == 1).sum()))

print("\n=== SIR ===")
for value in sorted(np.unique(sir[~np.isnan(sir)])):
    print(f"{value:.1f} dB:", int((sir == value).sum()))

print("\n=== DOSYA ===")
print("Farkli file_id:", len(np.unique(file_id)))

print("\nKontrol tamamlandi.")