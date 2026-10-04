| Model | Girdi | Doğruluk | F1 | ROC-AUC | Çıkarım (ms/örnek) | Parametre | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (enerji esigi) | Ham IQ gucu | 0.783 | 0.864 | 0.757 | 2.02 | 0 | 1 |
| 1D CNN | Ham IQ (I, Q) | 0.975 ± 0.029 | 0.982 ± 0.021 | 1.000 ± 0.000 | 145.48 ± 17.23 | 85,730 | 3 |
| 2D CNN | STFT spektrogrami | 0.886 ± 0.024 | 0.923 ± 0.013 | 0.961 ± 0.019 | 92.37 ± 0.29 | 60,706 | 3 |
