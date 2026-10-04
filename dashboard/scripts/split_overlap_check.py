import pandas as pd

s = pd.read_csv(r"data\dataset_index_split.csv")

splits = {}

for split in ["train", "val", "test"]:
    splits[split] = set(s.loc[s["split"] == split, "file"])

print("Train files:", len(splits["train"]))
print("Val files:", len(splits["val"]))
print("Test files:", len(splits["test"]))

print("Train-Val overlap:", len(splits["train"] & splits["val"]))
print("Train-Test overlap:", len(splits["train"] & splits["test"]))
print("Val-Test overlap:", len(splits["val"] & splits["test"]))