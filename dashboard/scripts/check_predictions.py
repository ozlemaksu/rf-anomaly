from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

required = ["y_true", "y_pred", "y_prob", "sir_db"]

files = [
    BASE / "data" / "cnn1d_predictions.csv",
    BASE / "data" / "cnn2d_predictions.csv"
]

for file in files:
    print(f"\nKontrol: {file.name}")

    if not file.exists():
        print("Dosya henüz yok.")
        continue

    df = pd.read_csv(file)

    print("Kayıt sayısı:", len(df))
    print("Sütunlar:", list(df.columns))

    missing = [col for col in required if col not in df.columns]

    if missing:
        print("EKSIK SÜTUNLAR:", missing)
    else:
        print("Format OK.")