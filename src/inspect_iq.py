import numpy as np, sys
p = sys.argv[1]
b = open(p, "rb").read()
print("boyut", len(b), "bayt; ilk 16 bayt:", b[:16].hex())
with np.errstate(all="ignore"):
    for dt in ("<f4", ">f4", "<f8", ">f8", "<i2", ">i2"):
        for off in (0, 8):
            n = (len(b) - off) // np.dtype(dt).itemsize
            a = np.frombuffer(b, dtype=dt, count=n, offset=off).astype(np.float64)
            print(f"{dt} ofset {off}: n={n} min={np.nanmin(a):.3g} max={np.nanmax(a):.3g} std={np.nanstd(a):.3g} nan={int(np.isnan(a).sum())}")
