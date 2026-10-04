# RF Anomali Tespiti — DSP → ML Teslim Paketi

## 1. Paket Amacı

Bu klasör, RF Anomali Tespiti projesinin DSP aşamasında işlenen IQ sinyallerinin ML/CNN aşamasında kullanılmak üzere hazırlanmış halini içerir.

DSP pipeline:

```text
Ham IQ Sinyali
      ↓
IQ Okuma
      ↓
Pencereleme
      ↓
STFT
      ↓
Magnitude → dB
      ↓
Normalizasyon
      ↓
256 × 512 Spectrogram
      ↓
ML/CNN
```

---

## 2. Dataset Özeti

Orijinal dataset:

- Toplam kayıt: **408**
- OnlyLTE kayıtları: **120**
- LTE+DSSS kayıtları: **288**

Her kayıt 4 pencereye ayrılmıştır.

Bunun sonucunda:

- OnlyLTE spectrogram: **480**
- LTE+DSSS spectrogram: **1152**
- Toplam spectrogram: **1632**

---

## 3. Label Sistemi

| Label | Sinyal tipi | Anlam |
|---|---|---|
| `0` | OnlyLTE | Normal |
| `1` | LTE+DSSS | Anomali |

ML modelinin hedef değişkeni `metadata.csv` içerisindeki `label` sütunudur.

---

## 4. Pencereleme

Her IQ kaydından **4 adet pencere** oluşturulmuştur.

Pencere uzunluğu:

```text
131072 complex IQ sample
```

Pencereler kayıt boyunca eşit şekilde dağıtılmıştır.

Kullanılan yaklaşım:

```python
np.linspace(
    0,
    signal_length - 131072,
    4,
    dtype=int
)
```

Bu nedenle 4 pencere birbirinin hemen arkasında olmak zorunda değildir; kaydın farklı bölümlerini temsil eder.

---

## 5. STFT Parametreleri

ML datasetinin oluşturulmasında aşağıdaki STFT parametreleri kullanılmıştır:

| Parametre | Değer |
|---|---|
| Pencere fonksiyonu | Hann |
| NFFT | 256 |
| Hop length | 256 |
| Overlap | 0 |
| Boundary | None |
| Padding | False |
| Spectrum | Two-sided |
| FFT shift | True |

Her pencerenin çıktısı:

```text
256 × 512
```

boyutundadır.

---

## 6. Frekans Çözünürlüğü

Frekans çözünürlüğü örnekleme frekansına bağlıdır:

```text
Δf = Fs / 256
```

Dataset içerisinde farklı sample rate değerleri bulunmaktadır.

| Sample rate | Frekans çözünürlüğü |
|---:|---:|
| 7.68 MHz | 30 kHz |
| 15.36 MHz | 60 kHz |
| 30.72 MHz | 120 kHz |

Bu nedenle aynı spectrogram pikseli bütün kayıtlar için aynı mutlak frekansa karşılık gelmez.

---

## 7. Zaman Çözünürlüğü

STFT hop length değeri 256 sample'dır.

```text
Δt = 256 / Fs
```

Yaklaşık zaman adımları:

| Sample rate | Zaman adımı |
|---:|---:|
| 7.68 MHz | 33.3 µs |
| 15.36 MHz | 16.7 µs |
| 30.72 MHz | 8.33 µs |

---

## 8. Spectrogram Oluşturma

STFT sonucunda kompleks spektrumun magnitude değeri alınmıştır:

```text
|STFT|
```

Daha sonra dB dönüşümü uygulanmıştır:

```text
20 × log10(|STFT| + 1e-12)
```

Ardından dB değerleri normalize edilmiştir.

Kullanılan aralık:

```text
-50 dB → 0
  0 dB → 1
```

Bu aralığın dışındaki değerler sınırlandırılmıştır.

Sonuç:

```text
0 ≤ spectrogram ≤ 1
```

Veri tipi:

```text
float32
```

---

## 9. Dosya Yapısı

Bu klasörün yapısı:

```text
ml_handoff/
│
├── spectrograms/
│   ├── OnlyLTE_frame_1_w0.npy
│   ├── OnlyLTE_frame_1_w1.npy
│   ├── ...
│   ├── Combined_LTE_DSSS_frame_264_w0.npy
│   └── ...
│
├── metadata.csv
├── dataset_config.json
└── README.md
```

Her `.npy` dosyası tek bir spectrogram içerir.

Örnek:

```text
Combined_LTE_DSSS_frame_264_w0.npy
```

İçeriği:

```text
shape = (256, 512)
dtype = float32
```

---

## 10. metadata.csv

`metadata.csv`, her spectrogram için gerekli bilgileri içerir.

Temel sütunlar:

```text
file
window
spectrogram
label
signal_type
sample_rate
snr_db
sir_db
```

Örneğin:

```text
file = Combined_LTE_DSSS_frame_264
window = 0
label = 1
signal_type = LTE_DSSS
sample_rate = 30720000
```

`spectrogram` sütunu ilgili `.npy` dosyasının yolunu gösterir.

---

## 11. ML Modeline Verilecek Girdi

CNN tarafında her spectrogram tek kanallı görüntü gibi değerlendirilebilir.

Temel giriş boyutu:

```text
256 × 512
```

CNN implementation'ına göre kanal boyutu eklenebilir:

```text
256 × 512 × 1
```

Buradaki `1`, spectrogramın tek kanal olduğunu ifade eder.

Spectrogramlar zaten `[0,1]` aralığında normalize edilmiştir.

---

## 12. Çok Önemli: Data Leakage

Train / validation / test ayrımı **spectrogram bazında yapılmamalıdır.**

Aynı recording içerisindeki 4 pencere birbirleriyle ilişkilidir:

```text
frame_264
├── w0
├── w1
├── w2
└── w3
```

Bu 4 pencerenin farklı dataset splitlerine dağıtılması data leakage oluşturabilir.

Örneğin aşağıdaki yapı kullanılmamalıdır:

```text
Train:
frame_264_w0
frame_264_w1

Test:
frame_264_w2
frame_264_w3
```

Bunun yerine recording bazlı split yapılmalıdır:

```text
Train:
frame_264 → w0,w1,w2,w3

Validation:
başka recordingler

Test:
başka recordingler
```

Yani aynı recordingin bütün pencereleri aynı split içerisinde bulunmalıdır.

---

## 13. Sınıf Dağılımı

Recording bazında:

```text
OnlyLTE       : 120
LTE+DSSS      : 288
```

Spectrogram bazında:

```text
OnlyLTE       : 480
LTE+DSSS      : 1152
```

Dolayısıyla dataset sınıfları dengeli değildir.

ML tarafında model eğitilirken class imbalance dikkate alınmalıdır.

Değerlendirmede yalnızca accuracy kullanılmaması; precision, recall, F1-score ve confusion matrix gibi metriklerin de incelenmesi önerilir.

---

## 14. Sample Rate Hakkında Önemli Not

Dataset içerisinde:

```text
7.68 MHz
15.36 MHz
30.72 MHz
```

sample rate değerleri bulunmaktadır.

Bu nedenle modelin farklı sample rate'lerdeki spectrogramları nasıl öğrendiği ayrıca kontrol edilmelidir.

`metadata.csv` içerisindeki:

```text
sample_rate
```

alanı bu amaçla kullanılabilir.

---

## 15. DSP Tarafında Yapılan Ek Analizler

Ana CNN inputu spectrogramdır.

Bunun yanında DSP aşamasında aşağıdaki analizler gerçekleştirilmiştir:

### FFT

Frekans domenindeki temel sinyal yapısı incelenmiştir.

### Band Ratio

Sinyal bandındaki ve dışındaki enerji oranı incelenmiştir.

Band ratio değerinin özellikle LTE+DSSS kayıtlarında CNN çıktısıyla ilişkili olduğu gözlemlenmiştir.

Bu sonuç, band ratio'nun potansiyel bir yardımcı özellik olabileceğini göstermektedir; tek başına anomali kanıtı olarak değerlendirilmemelidir.

### Gap Analysis

Spectrogram içerisindeki geçici enerji düşüşleri ve sinyal boşlukları incelenmiştir.

Bazı CNN hata örneklerinde belirgin gap yapıları görülmüştür.

Ancak bütün CNN hatalarını yalnızca gap yapısıyla açıklamak mümkün değildir.

### Autocorrelation

OnlyLTE ve LTE+DSSS örneklerinde otokorelasyon incelenmiştir.

İncelenen örneklerde belirgin ve düzenli bir tekrar yapısı gözlemlenmemiştir.

### Cyclostationary / SCF Analizi

SCF benzeri coherence tabanlı analiz gerçekleştirilmiştir.

Bazı LTE+DSSS kayıtlarında belirli cyclic-frequency yapılarının OnlyLTE kayıtlarına göre daha belirgin olduğu görülmüştür.

Dataset genelindeki deneysel özelliklerde:

```text
best_spacing_score
```

LTE+DSSS ve OnlyLTE sınıfları arasında belirgin ayrım göstermiştir.

Ancak kullanılan estimator araştırma amaçlı bir coherence tabanlı yaklaşımdır. Bu sonuç, tek başına belirli bir fiziksel DSSS imzasının kesin kanıtı olarak değerlendirilmemelidir.

SCF analizi ana CNN inputunun bir parçası değildir.

---

## 16. CNN Hata Analizi

Önceki CNN sonuçlarında bazı hataların belirli koşullarda yoğunlaştığı görülmüştür.

Özellikle:

- OnlyLTE false positive örnekleri çoğunlukla düşük SNR koşullarında görülmüştür.
- LTE+DSSS false negative örnekleri özellikle SIR = 10 dB koşullarında görülmüştür.
- Bazı false negative örneklerinde spectrogram görsel olarak temiz görünmesine rağmen model anomalinin varlığını algılayamamıştır.

Bu nedenle model performansı değerlendirilirken:

```text
SNR
SIR
Sample rate
Recording
```

gibi metadata alanlarının ayrıca incelenmesi önemlidir.

---

## 17. Tekrar Üretilebilirlik

DSP pipeline kodları:

```text
src/dsp/io.py
src/dsp/preprocessing.py
src/dsp/stft.py
src/dsp/pipeline.py
src/dsp/build_ml_dataset.py
```

Dataset oluşturma:

```text
python -m src.dsp.build_ml_dataset
```

Kullanılan tüm temel parametreler ayrıca:

```text
dataset_config.json
```

dosyasında saklanmaktadır.

---

## 18. ML Ekibine Teslim Edilen Nihai Çıktı

ML tarafının temel olarak kullanması gereken dosyalar:

```text
spectrograms/
metadata.csv
dataset_config.json
README.md
```

Ana ML girdisi:

```text
spectrograms/*.npy
```

Etiket ve metadata:

```text
metadata.csv
```

DSP parametreleri:

```text
dataset_config.json
```

Pipeline açıklaması ve kullanım kuralları:

```text
README.md
```

---

## 19. Kısa Özet

```text
408 IQ recording
        ↓
4 window / recording
        ↓
1632 spectrogram
        ↓
256 × 512
        ↓
float32
        ↓
[0,1]
        ↓
CNN / ML
```

Label:

```text
0 → OnlyLTE
1 → LTE+DSSS
```

En önemli veri bölme kuralı:

```text
Recording bazlı train / validation / test split
```

Aynı recordingin farklı pencereleri farklı splitlerde kullanılmamalıdır.