import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRE = DATA / "preprocessed"

errors = []

print("=" * 60)
print("FINAL INTEGRITY CHECK")
print("=" * 60)

# --------------------------------------------------
# 1. Dosyaları yükle
# --------------------------------------------------

file_index = pd.read_csv(PRE / "file_index.csv")
dataset_split = pd.read_csv(DATA / "dataset_index_split.csv")

X_iq = np.load(PRE / "X_iq.npy")
X_spec = np.load(PRE / "X_spec.npy")
y = np.load(PRE / "y.npy")
sir = np.load(PRE / "sir_db.npy")

frozen = np.load(
    PRE / "frozen_window_indices.npz"
)

train_idx = frozen["train_idx"]
val_idx = frozen["val_idx"]
test_idx = frozen["test_idx"]

print("\n[1] Dosya boyutları")

print("file_index:", len(file_index))
print("dataset_index_split:", len(dataset_split))
print("X_iq:", X_iq.shape)
print("X_spec:", X_spec.shape)
print("y:", y.shape)
print("sir:", sir.shape)

# --------------------------------------------------
# 2. Temel boyut kontrolü
# --------------------------------------------------

n = len(y)

if X_iq.shape[0] != n:
    errors.append("X_iq ve y örnek sayısı uyuşmuyor.")

if X_spec.shape[0] != n:
    errors.append("X_spec ve y örnek sayısı uyuşmuyor.")

if len(sir) != n:
    errors.append("sir_db ve y örnek sayısı uyuşmuyor.")

if n != 1632:
    errors.append(f"Beklenen 1632 window yerine {n} bulundu.")

# --------------------------------------------------
# 3. Frozen index kontrolü
# --------------------------------------------------

print("\n[2] Frozen index kontrolü")

all_idx = np.concatenate([
    train_idx,
    val_idx,
    test_idx
])

print("Train:", len(train_idx))
print("Val:", len(val_idx))
print("Test:", len(test_idx))
print("Toplam:", len(all_idx))

if len(np.unique(train_idx)) != len(train_idx):
    errors.append("Train indexlerinde duplicate var.")

if len(np.unique(val_idx)) != len(val_idx):
    errors.append("Validation indexlerinde duplicate var.")

if len(np.unique(test_idx)) != len(test_idx):
    errors.append("Test indexlerinde duplicate var.")

if len(np.unique(all_idx)) != n:
    errors.append("Train/Val/Test indexleri tam kapsam sağlamıyor.")

if set(train_idx) & set(val_idx):
    errors.append("Train-Val overlap var.")

if set(train_idx) & set(test_idx):
    errors.append("Train-Test overlap var.")

if set(val_idx) & set(test_idx):
    errors.append("Val-Test overlap var.")

# --------------------------------------------------
# 4. Label eşleşmesi
# --------------------------------------------------

print("\n[3] Label kontrolü")

file_ids = np.arange(n) // 4

expected_y = file_index.iloc[file_ids]["y"].to_numpy()

if not np.array_equal(y, expected_y):
    errors.append(
        "y.npy ile file_index.csv label değerleri uyuşmuyor."
    )

else:
    print("y.npy <-> file_index.csv: OK")

# --------------------------------------------------
# 5. SIR eşleşmesi
# --------------------------------------------------

print("\n[4] SIR kontrolü")

expected_sir = file_index.iloc[file_ids]["sir_db"].to_numpy()

sir_equal = np.isclose(
    np.nan_to_num(sir, nan=-9999),
    np.nan_to_num(expected_sir, nan=-9999)
)

if not np.all(sir_equal):
    errors.append(
        "sir_db.npy ile file_index.csv SIR değerleri uyuşmuyor."
    )

else:
    print("sir_db.npy <-> file_index.csv: OK")

# --------------------------------------------------
# 6. Prediction CSV kontrolü
# --------------------------------------------------

print("\n[5] Prediction CSV kontrolü")

prediction_files = [
    "cnn1d_s1_test_preds.csv",
    "cnn1d_s2_test_preds.csv",
    "cnn1d_s3_test_preds.csv",
    "cnn2d_s1_test_preds.csv",
    "cnn2d_s2_test_preds.csv",
    "cnn2d_s3_test_preds.csv"
]

for filename in prediction_files:

    path = DATA / filename

    if not path.exists():
        errors.append(
            f"{filename} bulunamadı."
        )
        continue

    df = pd.read_csv(path)

    print(
        filename,
        "rows =", len(df),
        "unique idx =", df["idx"].nunique()
    )

    if len(df) != len(test_idx):
        errors.append(
            f"{filename}: test row sayısı yanlış."
        )

    if set(df["idx"]) != set(test_idx):
        errors.append(
            f"{filename}: idx değerleri frozen test ile uyuşmuyor."
        )

    expected_labels = file_index.iloc[
        df["idx"].to_numpy() // 4
    ]["y"].to_numpy()

    if not np.array_equal(
        df["y_true"].to_numpy(),
        expected_labels
    ):
        errors.append(
            f"{filename}: y_true label'ları uyuşmuyor."
        )

# --------------------------------------------------
# 7. Sonuç
# --------------------------------------------------

print("\n" + "=" * 60)

if len(errors) == 0:

    print("SONUÇ: TÜM KONTROLLER BAŞARILI")
    print("=" * 60)

    print("\nVeri pipeline'ında tespit edilen hata yok.")

else:

    print("SONUÇ: PROBLEM BULUNDU")
    print("=" * 60)

    for error in errors:
        print("-", error)