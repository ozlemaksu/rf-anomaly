from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
METADATA = BASE / "data" / "MATLAB_Dataset" / "Metadata"

rows = []

for file in METADATA.glob("*.csv"):
    meta = pd.read_csv(file)

    signal_type = str(meta.iloc[0]["Signal_Type"])

    if signal_type == "LTE":
        label = 0
        sir_db = None
        prefix = "OnlyLTE_frame_"
    elif signal_type == "LTE_DSSS":
        label = 1
        sir_db = float(meta.iloc[0]["LTE_DSSS_SIR_dB"])
        prefix = "Combined_LTE_DSSS_frame_"
    else:
        continue

    frame = int(file.stem.replace(prefix, ""))

    rows.append({
        "file": file.name,
        "frame": frame,
        "label": label,
        "sir_db": sir_db,
        "signal_type": signal_type
    })

df = pd.DataFrame(rows)
df["group"] = "frame_" + df["frame"].astype(str)

output = BASE / "data" / "dataset_index.csv"
df.to_csv(output, index=False)

print(f"Toplam kayıt: {len(df)}")
print(df["label"].value_counts().sort_index())
print("\nSIR dağılımı:")
print(df[df["label"] == 1]["sir_db"].value_counts().sort_index())
print(f"\nOluşturuldu: {output}")