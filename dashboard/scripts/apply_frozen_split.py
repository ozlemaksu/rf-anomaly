import json
import numpy as np
import pandas as pd

DATA = "data/preprocessed"
INDEX_FILE = "data/dataset_index_split.csv"
FROZEN_FILE = "data/frozen_indices.json"

print("=== FROZEN SPLIT UYGULAMA ===")

index_df = pd.read_csv(INDEX_FILE)

with open(FROZEN_FILE, "r") as f:
    frozen = json.load(f)

X_file_index = pd.read_csv(DATA + "/file_index.csv")
file_id = np.load(DATA + "/file_id.npy")

print("Preprocess dosya sayisi:", len(X_file_index))
print("Frozen index kayit sayisi:", len(index_df))
print("Pencere sayisi:", len(file_id))

# Dosya adini split bilgisiyle eslestir
split_map = dict(zip(index_df["file"], index_df["split"]))

# preprocess file_index.csv dosya adlari .csv uzantisi icermiyor
split_map_no_ext = {}

for name, split in split_map.items():
    clean_name = str(name).replace(".csv", "")
    split_map_no_ext[clean_name] = split

# Her file_id icin split belirle
file_splits = []

for _, row in X_file_index.iterrows():
    filename = str(row["file"])

    if filename not in split_map_no_ext:
        raise ValueError("Split bulunamadi: " + filename)

    file_splits.append(split_map_no_ext[filename])

file_splits = np.array(file_splits)

# Her pencerenin splitini file_id'den al
window_split = file_splits[file_id]

train_idx = np.where(window_split == "train")[0]
val_idx = np.where(window_split == "val")[0]
test_idx = np.where(window_split == "test")[0]

print("\nPencere dagilimi:")
print("Train:", len(train_idx))
print("Val:", len(val_idx))
print("Test:", len(test_idx))

print("\nDosya dagilimi:")
print("Train:", np.sum(file_splits == "train"))
print("Val:", np.sum(file_splits == "val"))
print("Test:", np.sum(file_splits == "test"))

# Kontrol
print("\n=== SIZINTI KONTROLU ===")

train_files = set(file_id[train_idx].tolist())
val_files = set(file_id[val_idx].tolist())
test_files = set(file_id[test_idx].tolist())

print("Train-Val ortak:", len(train_files & val_files))
print("Train-Test ortak:", len(train_files & test_files))
print("Val-Test ortak:", len(val_files & test_files))

if train_files & val_files or train_files & test_files or val_files & test_files:
    raise ValueError("DOSYA SIZINTISI VAR!")

print("Dosya sizintisi yok.")

# Frozen split ile pencere indekslerini kaydet
output = DATA + "/frozen_window_indices.npz"

np.savez(
    output,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

print("\nOlusturuldu:")
print(output)