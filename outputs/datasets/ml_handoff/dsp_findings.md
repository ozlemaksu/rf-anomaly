# RF Anomali Tespiti — DSP Bulguları

## 1. Amaç

DSP aşamasının amacı, ham IQ sinyallerini ML modelinin kullanabileceği standart bir zaman-frekans gösterimine dönüştürmek ve LTE ile LTE+DSSS sinyalleri arasındaki olası ayırt edici sinyal özelliklerini incelemektir.

Ana ML girdisi spectrogramdır.

Ek DSP analizleri ise model hatalarını ve sinyal davranışını anlamak amacıyla gerçekleştirilmiştir.

---

# 2. Ana DSP Pipeline

Kullanılan ana akış:

```text
Raw IQ
 ↓
IQ Loader
 ↓
4 × 131072 sample window
 ↓
STFT
 ↓
Magnitude
 ↓
dB
 ↓
[-50, 0] dB clipping
 ↓
[0, 1] normalization
 ↓
256 × 512 spectrogram
 ↓
CNN
```

Her recordingden 4 adet pencere çıkarılmıştır.

Toplam:

- 408 recording
- 1632 spectrogram

---

# 3. IQ Sinyal Analizi

IQ dosyalarının ham binary formatı incelenmiştir.

Bir örnek dosyada:

- Header sonrasında float32 veriler bulunmaktadır.
- I ve Q örnekleri interleaved olarak saklanmaktadır.
- I ve Q ayrılarak kompleks IQ sinyali oluşturulmaktadır.

Temel gösterim:

```text
IQ = I + jQ
```

Örnek bir kayıtta:

- 614400 complex sample
- Sample rate = 7.68 MHz

olarak doğrulanmıştır.

---

# 4. FFT Analizi

FFT ile sinyalin frekans domenindeki genel yapısı incelenmiştir.

Örnek analiz:

```text
NFFT = 4096
Fs = 7.68 MHz
```

Frekans çözünürlüğü:

```text
Δf = Fs / NFFT
```

Bu örnekte:

```text
Δf = 1875 Hz
```

FFT sonucunda sinyal enerjisinin belirli bir frekans bandında yoğunlaştığı görülmüştür.

FFT temel frekans yapısını anlamak için kullanılmıştır; ana CNN girdisi olarak kullanılmamıştır.

---

# 5. STFT / Spectrogram

Ana ML inputu STFT tabanlı spectrogramdır.

Kullanılan parametreler:

| Parametre | Değer |
|---|---|
| Window | Hann |
| NFFT | 256 |
| Hop | 256 |
| Overlap | 0 |
| Boundary | None |
| Padding | False |
| Spectrum | Two-sided |
| FFT shift | True |

Her pencerenin sonucu:

```text
256 × 512
```

boyutundadır.

---

# 6. Normalizasyon

STFT magnitude değeri dB'e çevrilmiştir:

```text
20 × log10(|STFT| + 1e-12)
```

Daha sonra:

```text
-50 dB → 0
0 dB   → 1
```

aralığında normalize edilmiştir.

Aralık dışındaki değerler clipping ile sınırlandırılmıştır.

Sonuç:

```text
0 ≤ X ≤ 1
```

ve:

```text
dtype = float32
```

---

# 7. STFT Parametre Deneyleri

Farklı FFT boyutlarının CNN hata örnekleri üzerindeki etkisi incelenmiştir.

İncelenen değerler:

```text
NFFT = 128
NFFT = 256
NFFT = 512
```

### NFFT = 128

Daha iyi zaman çözünürlüğü elde edilmiştir.

Ancak incelenen false negative örneklerinde yeni bir DSSS yapısı ortaya çıkmamıştır.

### NFFT = 256

Ana ML pipeline için kullanılmıştır.

Zaman-frekans çözünürlüğü ile CNN input boyutu arasında uygun bir denge sağlamaktadır.

### NFFT = 512

Frekans çözünürlüğü artmıştır.

Ancak incelenen false negative örneklerinde 256 çözünürlüğünde görünmeyen belirgin bir DSSS yapısı tespit edilmemiştir.

### Sonuç

Basit NFFT değişiminin tek başına CNN'in zorlandığı örnekleri açıklamadığı görülmüştür.

Bu nedenle ana pipeline:

```text
NFFT = 256
```

olarak sabitlenmiştir.

---

# 8. Band Ratio Analizi

Sinyalin belirli frekans bölgelerindeki enerji dağılımını incelemek amacıyla band ratio özelliği hesaplanmıştır.

Genel olarak:

```text
band ratio =
in-band energy / out-of-band energy
```

şeklinde değerlendirilmiştir.

Bazı gruplarda band ratio ile CNN'in anomali olasılığı arasında belirgin monotonic ilişkiler görülmüştür.

Özellikle LTE+DSSS sınıfında ilişki daha belirgindir.

Ancak bu sonuç:

- band ratio'nun doğrudan DSSS tespit ettiği,
- veya CNN kararının sebebinin kesin olarak band ratio olduğu

anlamına gelmez.

Bu özellik daha çok açıklayıcı/yardımcı DSP feature olarak değerlendirilmelidir.

---

# 9. Gap Analysis

Spectrogram zaman ekseni üzerinde sinyal enerjisinin geçici olarak düştüğü bölgeler incelenmiştir.

Kullanılan temel özellikler:

```text
gap_frac
long_gap_frac
max_gap_ms
n_long_gaps
contrast_db
```

Bazı LTE+DSSS false negative örneklerinde belirgin gap yapıları görülmüştür.

Örneğin bazı pencerelerde:

- sinyal enerjisi uzun süre düşmekte,
- LTE bandı zayıflamakta,
- CNN anomali olasılığı düşmektedir.

Ancak bütün false negative örneklerinde aynı davranış görülmemiştir.

Bazı false negative pencereleri görsel olarak temiz ve sürekli bir LTE bandına sahiptir.

Dolayısıyla gap feature tek başına CNN hatalarını açıklamamaktadır.

---

# 10. Autocorrelation

Autocorrelation analizi, sinyal içerisinde belirgin periyodik tekrarların olup olmadığını incelemek amacıyla gerçekleştirilmiştir.

İncelenen örnekler:

```text
OnlyLTE_frame_100
Combined_LTE_DSSS_frame_264
```

Her iki örnekte de:

- lag = 0 civarında ana peak,
- hızlı decay,
- belirgin düzenli tekrar yapısının olmaması

gözlemlenmiştir.

Bu iki örnek üzerinde basit autocorrelation yaklaşımı belirgin bir sınıf ayrımı göstermemiştir.

Bu nedenle autocorrelation ana ML feature'ı olarak kullanılmamıştır.

---

# 11. Cyclostationary / SCF Analizi

Cyclostationary özellikleri araştırmak amacıyla coherence tabanlı bir SCF-benzeri analiz gerçekleştirilmiştir.

Amaç:

```text
LTE
vs
LTE+DSSS
```

sinyallerinde cyclic-frequency eksenindeki farklılıkları incelemektir.

İncelenen 30.72 MHz sample-rate kayıtlarında bazı LTE+DSSS örneklerinde:

- non-zero cyclic frequency bölgelerinde düzenli yapılar,
- belirli alpha aralıklarında tekrarlayan enerji,
- OnlyLTE örneklerinde daha homojen yapı

gözlemlenmiştir.

---

# 12. SCF Feature Validation

204 adet 30.72 MHz recording üzerinde deneysel SCF feature validation yapılmıştır:

```text
OnlyLTE      = 60
LTE+DSSS     = 144
```

İncelenen özelliklerden bazıları:

```text
target_1p2mhz_comb_score
target_energy_ratio
best_spacing_score
max_to_median
```

Sonuçlar:

| Feature | AUC | Mann-Whitney p |
|---|---:|---:|
| target_1p2mhz_comb_score | 0.682 | 4.21 × 10⁻⁵ |
| target_energy_ratio | 0.681 | 4.66 × 10⁻⁵ |
| best_spacing_score | 0.903 | 1.41 × 10⁻¹⁹ |
| max_to_median | 0.791 | 6.16 × 10⁻¹¹ |

Bu sonuçlar, kullanılan deneysel SCF özelliklerinin LTE ve LTE+DSSS kayıtları arasında istatistiksel ayrım taşıyabildiğini göstermektedir.

Ancak önemli bir metodolojik sınırlama vardır:

**Bu estimator textbook anlamında tam ve rigoröz bir SCF implementasyonu değildir.**

Dolayısıyla bu sonuçlar:

- "DSSS'nin kesin fiziksel imzası bulundu"
- "1.2 MHz kesin olarak DSSS spreading frekansıdır"

şeklinde yorumlanmamalıdır.

Daha doğru yorum:

> Kullanılan coherence tabanlı cyclic-frequency özellikleri, LTE+DSSS sınıfında OnlyLTE sınıfına göre daha belirgin periyodik/structured enerji davranışı göstermiş ve bazı feature'lar sınıfları istatistiksel olarak ayırabilmiştir.

SCF analizi bu proje kapsamında:

```text
araştırma / açıklama / yardımcı feature
```

olarak değerlendirilmelidir.

Ana CNN inputu değildir.

---

# 13. CNN Hata Analizi

CNN sonuçları incelendiğinde hata örneklerinin bazı metadata koşullarıyla ilişkili olduğu görülmüştür.

## False Positive

OnlyLTE kayıtlarının yanlışlıkla LTE+DSSS olarak sınıflandırıldığı örnekler özellikle düşük SNR koşullarında görülmüştür.

Örneğin:

```text
OnlyLTE_frame_2
OnlyLTE_frame_7
OnlyLTE_frame_4
```

gibi bazı 0 dB SNR kayıtlarında false positive örnekleri görülmüştür.

---

# 14. False Negative

LTE+DSSS kayıtlarının normal LTE olarak sınıflandırıldığı örnekler özellikle:

```text
SIR = 10 dB
```

koşulunda yoğunlaşmıştır.

Örnek:

```text
Combined_LTE_DSSS_frame_264
```

Bu kayıt üzerinde bazı pencerelerde belirgin gap yapıları görülürken bazı pencereler görsel olarak temiz LTE bandına benzemektedir.

Bu durum önemli bir gözlemdir:

> Anomali sinyalinin varlığı her zaman spectrogram üzerinde görsel olarak belirgin bir ekstra enerji bölgesi şeklinde ortaya çıkmayabilir.

---

# 15. Sample Rate Etkisi

Dataset içerisinde:

```text
7.68 MHz
15.36 MHz
30.72 MHz
```

sample rate değerleri bulunmaktadır.

CNN hata analizi sırasında sample rate ile hata dağılımı arasında farklılıklar gözlemlenmiştir.

Ancak sample rate ile diğer metadata parametreleri tamamen bağımsız değildir.

Bu nedenle sample rate etkisi tek başına nedensel bir sonuç olarak yorumlanmamalıdır.

ML tarafında sonuçların sample rate bazında ayrıca raporlanması faydalıdır.

---

# 16. Ana DSP Sonuçları

DSP çalışmasının ana sonuçları:

### 1.

Ham IQ sinyalleri başarılı şekilde standartlaştırılmıştır.

### 2.

Her recording 4 pencereye ayrılarak toplam:

```text
1632 × 256 × 512
```

spectrogram dataset'i oluşturulmuştur.

### 3.

STFT tabanlı spectrogram CNN için ana DSP inputu olarak belirlenmiştir.

### 4.

NFFT = 128 / 256 / 512 karşılaştırmasında incelenen hata örneklerinde yalnızca FFT çözünürlüğünü değiştirmek belirgin yeni DSSS yapısı ortaya çıkarmamıştır.

### 5.

Band ratio ve gap özellikleri bazı CNN davranışlarını açıklamaya yardımcı olmuştur.

### 6.

Basit autocorrelation incelenen örneklerde belirgin sınıf ayrımı göstermemiştir.

### 7.

Coherence tabanlı SCF analizi LTE+DSSS kayıtlarında bazı structured cyclic-frequency davranışlarını göstermiştir.

### 8.

SCF özellikleri dataset genelinde istatistiksel sınıf ayrımı gösterebilmiştir.

### 9.

CNN false positive ve false negative örnekleri özellikle SNR, SIR ve sample rate gibi koşullar dikkate alınarak incelenmelidir.

---

# 17. ML Ekibine Önerilen Kullanım

Ana model inputu:

```text
spectrograms/*.npy
```

Label:

```text
0 → OnlyLTE
1 → LTE+DSSS
```

Metadata:

```text
metadata.csv
```

Önemli:

**Train / validation / test split recording bazında yapılmalıdır.**

Aynı recordinge ait 4 pencere farklı splitlere dağıtılmamalıdır.

---

# 18. DSP Pipeline'ın Nihai Durumu

DSP tarafındaki ana pipeline artık dondurulmuştur:

```text
Raw IQ
 ↓
IQ Loader
 ↓
4 × 131072 windows
 ↓
Hann STFT
 ↓
NFFT = 256
 ↓
Hop = 256
 ↓
Two-sided FFT
 ↓
FFT shift
 ↓
Magnitude
 ↓
dB
 ↓
[-50, 0] dB normalization
 ↓
[0, 1]
 ↓
float32
 ↓
256 × 512
 ↓
ML / CNN
```

Bu pipeline'ın parametreleri değiştirilmemelidir.

Yeni DSP deneyleri yapılacaksa ana dataset pipeline'ından ayrı olarak gerçekleştirilmelidir.