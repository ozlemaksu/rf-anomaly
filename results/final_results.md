# Nihai sonuçlar (donmuş split, test = 240 pencere / 60 kayıt)

Eşik 0.5 (baseline: eğitimden gelen eşik 1.236). Üç seed: 1, 2, 3.

| Model | Doğruluk | F1 | ROC-AUC | FP | FN |
|---|---|---|---|---|---|
| CNN1D (3 seed ort. ± std) | 0.975 ± 0.029 | 0.982 ± 0.021 | 1.000 ± 0.000 | 0.0 ± 0.0 | 6.0 ± 7.0 |
| CNN1D (3 seed ensemble) | 0.992 | 0.994 | 1.000 | 0.0 | 2.0 |
| CNN2D (3 seed ort. ± std) | 0.886 ± 0.024 | 0.923 ± 0.013 | 0.961 ± 0.019 | 15.0 ± 11.5 | 12.3 ± 7.0 |
| CNN2D (3 seed ensemble) | 0.896 | 0.930 | 0.965 | 16.0 | 9.0 |
| Enerji esigi (baseline) | 0.783 | 0.864 | 0.757 | 41.0 | 11.0 |

## %95 güven aralığı (kayıt bazlı bootstrap, 2000 tekrar, 60 test kaydı)

| Model | Doğruluk | F1 | ROC-AUC |
|---|---|---|---|
| CNN1D ensemble | [0.975, 1.000] | [0.981, 1.000] | [0.999, 1.000] |
| CNN2D ensemble | [0.829, 0.954] | [0.879, 0.972] | [0.929, 0.991] |
| Enerji esigi | [0.683, 0.875] | [0.788, 0.927] | [0.626, 0.872] |

## Hata kırılımı (ensemble, eşik 0.5): OnlyLTE ve SNR

| SNR (dB) | OnlyLTE pencere | CNN1D FP | CNN2D FP | Baseline FP |
|---|---|---|---|---|
| 0 | 28 | 0 | 16 | 28 |
| 5 | 16 | 0 | 0 | 13 |
| 10 | 20 | 0 | 0 | 0 |

## Hata kırılımı: LTE+DSSS ve SIR

| SIR (dB) | LTE+DSSS pencere | CNN1D FN | CNN2D FN | Baseline FN |
|---|---|---|---|---|
| 0 | 76 | 0 | 0 | 0 |
| 5 | 44 | 0 | 0 | 0 |
| 10 | 56 | 2 | 9 | 11 |

## Model boyutu ve CPU gecikmesi (Colab CPU, batch=1, model only)

| Model | Parametre | CPU ms/örnek |
|---|---|---|
| CNN1D | 85,730 | 145.5 ± 17.2 |
| CNN2D | 60,706 | 92.4 ± 0.3 |
| Enerji eşiği | 0 | 2.0 |