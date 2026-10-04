import pandas as pd
from pathlib import Path

DATA = Path("data")

prediction_files = {
    "1D CNN": DATA / "cnn1d_predictions.csv",
    "2D CNN": DATA / "cnn2d_predictions.csv"
}

required_columns = [
    "y_true",
    "y_pred",
    "y_prob",
    "sir_db"
]

print("=== CNN SONUC ANALIZI ===")

for model_name, file_path in prediction_files.items():

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    if not file_path.exists():
        print("Prediction dosyasi henuz yok:")
        print(file_path)
        continue

    df = pd.read_csv(file_path)

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print("EKSIK SUTUNLAR:", missing)
        continue

    print("Kayit sayisi:", len(df))

    print("\nSutunlar:")
    print(list(df.columns))

print("\n=== ANALIZ TAMAMLANDI ===")