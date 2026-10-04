from pathlib import Path

DATA = Path("data")

folders = [
    DATA / "results",
    DATA / "plots",
    DATA / "models"
]

files = [
    DATA / "cnn1d_predictions.csv",
    DATA / "cnn2d_predictions.csv"
]

print("=== PROJE SONUCLARI KONTROL ===")

print("\nKlasorler:")

for folder in folders:
    if folder.exists():
        print("[OK]", folder)
    else:
        print("[YOK]", folder)

print("\nPrediction dosyalari:")

for file in files:
    if file.exists():
        print("[HAZIR]", file)
    else:
        print("[BEKLENIYOR]", file)

print("\nKontrol tamamlandi.")