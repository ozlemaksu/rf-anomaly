from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

INDEX_FILE = BASE / "data" / "dataset_index_split.csv"
FROZEN_FILE = BASE / "data" / "frozen_indices.json"

print("=== FINAL DATA CHECK ===\n")

# 1. Dosyaları kontrol et
if not INDEX_FILE.exists():
    print("HATA: dataset_index_split.csv bulunamadı.")
    raise SystemExit

if not FROZEN_FILE.exists():
    print("HATA: frozen_indices.json bulunamadı.")
    raise SystemExit

df = pd.read_csv(INDEX_FILE)

with open(FROZEN_FILE, "r", encoding="utf-8") as f:
    frozen = json.load(f)

# 2. Toplam kayıt
print("1) TOPLAM KAYIT")
print("Toplam:", len(df))

# 3. Sınıf dağılımı
print("\n2) SINIF DAĞILIMI")
print(df["label"].value_counts().sort_index())

# 4. SIR dağılımı
print("\n3) SIR DAĞILIMI")
print(
    df[df["label"] == 1]["sir_db"]
    .value_counts()
    .sort_index()
)

# 5. Split kayıt sayıları
print("\n4) SPLIT KAYITLARI")
print(df["split"].value_counts())

# 6. Split grup sayıları
print("\n5) SPLIT GRUPLARI")
group_counts = df.groupby("split")["group"].nunique()
print(group_counts)

# 7. Leakage kontrolü
print("\n6) LEAKAGE KONTROLÜ")

train_groups = set(
    df[df["split"] == "train"]["group"].astype(str)
)

val_groups = set(
    df[df["split"] == "val"]["group"].astype(str)
)

test_groups = set(
    df[df["split"] == "test"]["group"].astype(str)
)

train_val = train_groups & val_groups
train_test = train_groups & test_groups
val_test = val_groups & test_groups

print("Train ∩ Val:", len(train_val))
print("Train ∩ Test:", len(train_test))
print("Val ∩ Test:", len(val_test))

if not train_val and not train_test and not val_test:
    print("LEAKAGE YOK.")
else:
    print("DIKKAT: SPLITLER ARASINDA ORTAK GRUP VAR!")

# 8. Frozen index kontrolü
print("\n7) FROZEN INDEX KONTROLÜ")

frozen_train = set(frozen["train"])
frozen_val = set(frozen["val"])
frozen_test = set(frozen["test"])

print("Frozen train:", len(frozen_train))
print("Frozen val:", len(frozen_val))
print("Frozen test:", len(frozen_test))

if (
    train_groups == frozen_train
    and val_groups == frozen_val
    and test_groups == frozen_test
):
    print("Frozen index ile mevcut split AYNI.")
else:
    print("DIKKAT: Frozen index ile mevcut split FARKLI!")

# 9. SIR x split
print("\n8) SIR x SPLIT")
print(
    pd.crosstab(
        df["split"],
        df["sir_db"]
    )
)

# 10. Sonuç
print("\n=== KONTROL TAMAMLANDI ===")