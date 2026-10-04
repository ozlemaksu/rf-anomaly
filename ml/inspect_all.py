import numpy as np, pandas as pd, os
root = r"data\ham\sentetik\MATLAB_Dataset"
iqd = os.path.join(root, "IQ")
md = os.path.join(root, "Metadata")

def ac1(x):
    x = x.astype(np.float64)
    x = x - x.mean()
    return float((x[:-1] * x[1:]).mean() / (x * x).mean())

def layout(name):
    b = open(os.path.join(iqd, name), "rb").read()
    h = np.frombuffer(b, "<i4", 2)
    a = np.frombuffer(b, "<f4", offset=8)
    n = a.size // 2
    print(name, "baslik:", h.tolist(), "float sayisi:", a.size)
    print("  ardisik satir (I hepsi, sonra Q):  I-ac1 %.3f  Q-ac1 %.3f" % (ac1(a[:n]), ac1(a[n:])))
    print("  ic ice (I,Q,I,Q...):               I-ac1 %.3f  Q-ac1 %.3f" % (ac1(a[0::2]), ac1(a[1::2])))

layout("Combined_LTE_DSSS_frame_103")
layout("OnlyLTE_frame_1")

rows = []
for f in sorted(os.listdir(iqd)):
    h = np.frombuffer(open(os.path.join(iqd, f), "rb").read(8), "<i4")
    c = os.path.join(md, f + ".csv")
    if not os.path.exists(c):
        rows.append(dict(file=f, rows=int(h[0]), cols=int(h[1]), sig="CSV_YOK"))
        continue
    m = pd.read_csv(c).iloc[0]
    rows.append(dict(file=f, rows=int(h[0]), cols=int(h[1]), sig=m.get("Signal_Type"),
                     sr=m.get("LTE_SR"), snr=m.get("LTE_SNR_dB"), sir=m.get("LTE_DSSS_SIR_dB")))
df = pd.DataFrame(rows)
df.to_csv(r"data\file_table.csv", index=False)
print()
print(df.groupby(["sig", "sr", "cols"], dropna=False).size().to_string())
print()
print("SNR degerleri:", df["snr"].value_counts(dropna=False).to_dict())
