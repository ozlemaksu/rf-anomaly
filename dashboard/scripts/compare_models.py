import pandas as pd
from pathlib import Path

DATA = Path("data")

files = {
    "1D CNN": DATA / "cnn1d_predictions.csv",
    "2D CNN": DATA / "cnn2d_predictions.csv"
}

print("=== MODEL KARSILASTIRMA ===")

for model_name, file_path in files.items():

    print("\n" + "=" * 40)
    print(model_name)
    print("=" * 40)

    if not file_path.exists():
        print("Prediction dosyasi henuz yok.")
        continue

    df = pd.read_csv(file_path)

    required = [
        "y_true",
        "y_pred",
        "y_prob",
        "sir_db"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        print("EKSIK SUTUNLAR:", missing)
        continue

    print("Kayit sayisi:", len(df))
    print("Format: OK")

print("\n=== KARSILASTIRMA HAZIR ===")