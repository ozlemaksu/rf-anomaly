import streamlit as st
import pandas as pd
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
    layout="wide"
)

# =========================================================
# BAŞLIK
# =========================================================

st.title("📡 RF Anomali Tespit Sistemi")
st.caption("LTE ve LTE + DSSS sinyallerinin otomatik anomalilik tespiti")

st.info(
    "Veri sentetiktir ve test seti 60 kayıttan oluşur. "
    "Toplam 240 test penceresi değerlendirilmiştir. "
    "CNN1D seed ortalaması: %97.5 ± %2.9; "
    "3-seed ensemble: %99.2 doğruluk "
    "(kayıt düzeyi bootstrap %95 GA: 0.975–1.000)."
)

# =========================================================
# ANA MODEL SONUÇLARI
# =========================================================

st.header("🏆 Model Performansı")

performance = pd.DataFrame({
    "Model": [
        "CNN1D Ensemble",
        "CNN2D Ensemble",
        "Energy Baseline"
    ],
    "Accuracy": [
        0.9917,
        0.8958,
        0.7833
    ],
    "F1": [
        0.9943,
        0.9304,
        0.8639
    ],
    "ROC-AUC": [
        0.99973,
        0.96547,
        0.7568
    ],
    "FP": [
        0,
        16,
        41
    ],
    "FN": [
        2,
        9,
        11
    ]
})

st.dataframe(
    performance.style.format({
        "Accuracy": "{:.2%}",
        "F1": "{:.2%}",
        "ROC-AUC": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# ÖNE ÇIKAN CNN1D SONUCU
# =========================================================

st.subheader("CNN1D Ensemble — Test Sonucu")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "99.2%")
col2.metric("F1", "99.4%")
col3.metric("ROC-AUC", "0.9997")
col4.metric("False Positive", "0")

st.success(
    "Bu test setindeki 64 OnlyLTE penceresinde false positive gözlenmemiştir. "
    "Toplam 240 test penceresinde 2 false negative oluşmuştur."
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.header("🔎 CNN1D Ensemble — Confusion Matrix")

cm = pd.DataFrame(
    [
        ["OnlyLTE", 64, 0],
        ["LTE + DSSS", 2, 174]
    ],
    columns=[
        "Gerçek sınıf",
        "Normal tahmin",
        "Anomali tahmin"
    ]
)

st.dataframe(
    cm,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "TN = 64, FP = 0, FN = 2, TP = 174. "
    "Değerler test setindeki 240 pencereye aittir."
)

# =========================================================
# MODEL HATALARI
# =========================================================

st.header("⚠️ Model Hataları")

error_data = pd.DataFrame({
    "Model": [
        "CNN1D Ensemble",
        "CNN2D Ensemble",
        "Energy Baseline"
    ],
    "False Positive": [0, 16, 41],
    "False Negative": [2, 9, 11]
})

st.bar_chart(
    error_data.set_index("Model"),
    use_container_width=True,
    stack=False
)

st.caption(
    "Grafik 0'dan başlayan hata sayısını gösterir. "
    "False positive: normal LTE örneğinin anomalik sınıflandırılması. "
    "False negative: LTE + DSSS örneğinin normal sınıflandırılması."
)

# =========================================================
# SIR ANALİZİ
# =========================================================

st.header("📊 SIR Bazlı Hata Analizi")

sir = st.selectbox(
    "SIR değeri seçin",
    [0.0, 5.0, 10.0],
    index=2
)

sir_summary = pd.DataFrame({
    "SIR (dB)": [0.0, 5.0, 10.0],
    "CNN1D FN": [0, 0, 2],
    "CNN2D FN": [0, 0, 9]
})

selected_sir = sir_summary[
    sir_summary["SIR (dB)"] == sir
]

st.dataframe(
    selected_sir,
    use_container_width=True,
    hide_index=True
)

st.caption("Tüm SIR değerleri (test: 76 / 44 / 56 pencere, SIR 0 / 5 / 10 dB)")
st.dataframe(
    sir_summary,
    use_container_width=True,
    hide_index=True
)

if sir == 10.0:
    st.warning(
        "Bu test setinde hatalar SIR = 10 dB koşulunda yoğunlaşmıştır. "
        "Örnek sayısı sınırlı olduğu için bundan istatistiksel genelleme yapılmamalıdır."
    )
else:
    st.success(
        f"SIR = {sir:.0f} dB koşulunda CNN1D ensemble için "
        "false negative gözlenmemiştir."
    )

st.subheader("SNR Bazlı False Positive Analizi")

snr_fp = pd.DataFrame({
    "Model": [
        "CNN1D Ensemble",
        "CNN2D Ensemble"
    ],
    "SNR = 0 dB": [0, 16],
    "SNR = 5 dB": [0, 0],
    "SNR = 10 dB": [0, 0]
})

st.dataframe(
    snr_fp,
    use_container_width=True,
    hide_index=True
)

st.info(
    "Bu test setinde CNN2D ensemble'ın 16 false positive hatasının tamamı "
    "SNR = 0 dB koşulundaki normal OnlyLTE pencerelerinde oluşmuştur. "
    "CNN1D ensemble'da false positive gözlenmemiştir."
)

# =========================================================
# ENERGY BASELINE
# =========================================================

st.header("⚡ Energy Baseline")

energy_file = RESULTS / "energy_baseline_metrics.json"

if energy_file.exists():

    with open(
        energy_file,
        "r",
        encoding="utf-8"
    ) as f:
        energy = json.load(f)

    cols = st.columns(5)

    cols[0].metric(
        "Accuracy",
        "78.33%"
    )

    cols[1].metric(
        "F1",
        "86.39%"
    )

    cols[2].metric(
        "ROC-AUC",
        "0.7568"
    )

    cols[3].metric(
        "False Positive",
        "41"
    )

    cols[4].metric(
        "False Negative",
        "11"
    )

    st.write(
        "Energy threshold: **1.2356**"
    )

    st.caption(
        "Eşik, validation setinde F1'i en yüksek yapan değer olarak seçilir "
        "(ml/baseline_energy.py). 1.2356 değeri resmi koşunun kayıtlı sonuç dosyasından "
        "alınmıştır (ml/results/v2). Bağımsız bir yeniden çalıştırmada "
        "(dashboard/scripts/energy_baseline.py) eşik 1.2532, doğruluk %77.5 çıktı; "
        "ROC-AUC aynıdır (0.7568), fark yalnızca eşik arama yönteminden gelir."
    )

else:
    st.warning(
        "energy_baseline_metrics.json bulunamadı."
    )

# =========================================================
# GENEL KARŞILAŞTIRMA
# =========================================================

st.header("📈 Genel Karşılaştırma")

comparison_chart = pd.DataFrame({
    "Model": [
        "CNN1D Ensemble",
        "CNN2D Ensemble",
        "Energy Baseline"
    ],
    "Accuracy": [
        99.17,
        89.58,
        78.33
    ],
    "F1": [
        99.43,
        93.04,
        86.39
    ],
    "ROC-AUC": [
        99.973,
        96.547,
        75.68
    ]
})

st.bar_chart(
    comparison_chart.set_index("Model"),
    use_container_width=True,
    stack=False
)

st.caption(
    "Accuracy, F1 ve ROC-AUC yüzde ölçeğinde; "
    "metrikler yan yana karşılaştırılmaktadır."
)

# =========================================================
# LATENCY
# =========================================================

st.header("⏱️ Çıkarım Gecikmesi")

latency = pd.DataFrame({
    "Model": [
        "CNN1D",
        "CNN2D",
        "Energy"
    ],
    "Latency (ms)": [
        145.5,
        92.4,
        2.018
    ]
})

st.dataframe(
    latency.style.format({
        "Latency (ms)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "CNN1D tek model çıkarımında CNN2D'den daha yavaştır "
    "(145.5 ms vs 92.4 ms). "
    "CNN1D ensemble üç seed modeli çalıştırdığı için, "
    "tek model süresi üzerinden yaklaşık 3× model çıkarım maliyeti oluşur "
    "(ölçülmüş ensemble latency değildir). "
    "CNN ölçümleri model-only CPU inference içindir. "
    "CNN2D değerine STFT preprocessing dahil değildir. "
    "Energy ölçümü NumPy tabanlı tek pencere işlemini temsil eder; "
    "ölçüm koşulları CNN ölçümleriyle birebir aynı değildir."
)

# =========================================================
# SİNYAL GÖRSELLEŞTİRME
# =========================================================

st.header("📡 Sinyal Görselleştirme")

st.caption(
    "Örnek kayıt: Combined_LTE_DSSS | Etiket: Anomali | "
    "SNR: 10 dB | SIR: 0 dB"
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Raw I/Q")

    raw_iq = DATA / "raw_iq_example.png"

    if raw_iq.exists():
        st.image(
            raw_iq,
            use_container_width=True
        )
        st.caption(
            "Raw I/Q örneği — X: zaman örnekleri, Y: genlik"
        )
    else:
        st.warning(
            "raw_iq_example.png bulunamadı."
        )

with col2:

    st.subheader("STFT Spectrogram")

    stft = DATA / "stft_example.png"

    if stft.exists():
        st.image(
            stft,
            use_container_width=True
        )
        st.caption(
            "STFT spektrogramı — X: zaman penceresi, "
            "Y: frekans bini, değer: dB"
        )
    else:
        st.warning(
            "stft_example.png bulunamadı."
        )

# =========================================================
# MODEL AÇIKLAMALARI
# =========================================================

st.header("🧠 Kullanılan Yaklaşımlar")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("CNN1D")
    st.write(
        "Ham I/Q sinyalinden doğrudan zaman domeni özelliklerini öğrenir."
    )

with col2:
    st.subheader("CNN2D")
    st.write(
        "STFT spektrogramı üzerinden zaman-frekans özelliklerini öğrenir."
    )

with col3:
    st.subheader("Energy Baseline")
    st.write(
        "Sinyal gücünü/enerjisini eşik değer kullanarak sınıflandırır."
    )

# =========================================================
# SONUÇ
# =========================================================

st.header("📌 Sonuç")

st.markdown(
    """
**CNN1D ensemble**, bu test setinde en yüksek performansı göstermiştir.

- Accuracy: **%99.2**
- F1: **%99.4**
- ROC-AUC: **0.9997**
- False Positive: **0**
- False Negative: **2**

Bu sonuçlar **240 test penceresi ve 60 kayıt** üzerinden elde edilmiştir.
CNN1D seed sonuçlarının ortalaması **%97.5 ± %2.9 accuracy** düzeyindedir.
Ensemble sonucu test setinde performansı artırmıştır.

CNN2D ensemble'da görülen false positive hatalarının tamamı
bu test setindeki **SNR = 0 dB normal OnlyLTE** grubunda oluşmuştur.

Sonuçlar sentetik test verisine aittir ve gerçek dünya performansı veya
%100 güvenilirlik garantisi olarak yorumlanmamalıdır.
"""
)

st.caption(
    "RF Anomali Tespit Sistemi • CNN1D / CNN2D / Energy Baseline"
)