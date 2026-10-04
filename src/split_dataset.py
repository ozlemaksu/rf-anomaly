from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE / "data" / "dataset_index.csv")

groups = groups = df["group"].astype(str).unique().tolist()

# %70 train, %30 geçici
train_groups, temp_groups = train_test_split(
    groups,
    test_size=0.30,
    random_state=42
)

# Geçicinin yarısı validation, yarısı test
val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    random_state=42
)

df["split"] = "test"
df.loc[df["group"].isin(train_groups), "split"] = "train"
df.loc[df["group"].isin(val_groups), "split"] = "val"

output = BASE / "data" / "dataset_index_split.csv"
df.to_csv(output, index=False)

print("Split grup sayıları:")
print(df.groupby("split")["group"].nunique())

print("\nSplit kayıt sayıları:")
print(df["split"].value_counts())

print("\nDosya oluşturuldu:")
print(output)