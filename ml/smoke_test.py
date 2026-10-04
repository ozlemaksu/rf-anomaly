"""Duman testi: tum boru hattini sahte veriyle bastan sona dener. Tek komut:

    python ml/smoke_test.py

Adimlar: sahte veri uret -> veri kontrolu -> baseline -> 1D CNN -> 2D CNN -> asiri ogrenme -> tablo.
Gercek veri gelmeden ONCE bir kez calistirin; hepsi [ OK ] ise kodunuz hazirdir ve
gercek veride bir sorun cikarsa sebebi veridir.

Not: bu testin dogruluk rakamlari anlamsizdir (sahte veri). Onemli olan hatasiz bitmesidir.
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
FAKE = os.path.join("data", "fake")
OUT = os.path.join("ml", "results", "smoke")
X_IQ, X_SPEC = os.path.join(FAKE, "X_iq.npy"), os.path.join(FAKE, "X_spec.npy")
Y, SPL = os.path.join(FAKE, "y.npy"), os.path.join(FAKE, "splits.npz")

steps = []


def run(name, cmd, must_exist=(), capture=False):
    t0 = time.time()
    print(f"\n--- {name} ---", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture)
    out = (r.stdout or "") + (r.stderr or "") if capture else ""
    ok = r.returncode == 0
    missing = [p for p in must_exist if not os.path.exists(os.path.join(ROOT, p))]
    if ok and missing:
        ok = False
        print("beklenen cikti yok:", missing)
    if not ok and capture:
        print(out[-1500:])
    steps.append((name, ok, time.time() - t0))
    return ok, out


def main():
    try:
        import torch  # noqa: F401
        import sklearn  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as e:
        sys.exit(f"Eksik kutuphane: {e.name}. Kurun: pip install torch numpy pandas scikit-learn scipy")

    s = os.path.join("ml")
    run("1. sahte veri uret", [PY, f"{s}/make_fake_data.py", "--out", FAKE],
        must_exist=[X_IQ, X_SPEC, Y, SPL])
    run("2. veri sozlesmesi kontrolu", [PY, f"{s}/check_data.py", "--dir", FAKE])
    run("3. enerji baseline", [PY, f"{s}/baseline_energy.py", "--x", X_IQ, "--y", Y, "--splits", SPL,
                               "--out_dir", OUT], must_exist=[f"{OUT}/baseline_energy_metrics.json"])
    run("4. 1D CNN (3 epoch)", [PY, f"{s}/train.py", "--model", "cnn1d", "--x", X_IQ, "--y", Y,
                                "--splits", SPL, "--epochs", "3", "--out_dir", OUT, "--tag", "cnn1d_s1"],
        must_exist=[f"{OUT}/cnn1d_s1_metrics.json", f"{OUT}/cnn1d_s1_test_preds.csv"])
    run("5. 2D CNN (3 epoch)", [PY, f"{s}/train.py", "--model", "cnn2d", "--x", X_SPEC, "--y", Y,
                                "--splits", SPL, "--epochs", "3", "--out_dir", OUT, "--tag", "cnn2d_s1"],
        must_exist=[f"{OUT}/cnn2d_s1_metrics.json", f"{OUT}/cnn2d_s1_test_preds.csv"])

    ok, out = run("6. asiri ogrenme testi (32 ornek, 1D CNN)",
                  [PY, f"{s}/train.py", "--model", "cnn1d", "--x", X_IQ, "--y", Y, "--splits", SPL,
                   "--subset", "32", "--epochs", "80", "--patience", "1000",
                   "--out_dir", OUT, "--tag", "overfit"], capture=True)
    if ok:
        losses = [float(v) for v in re.findall(r"train_loss ([0-9.]+)", out)]
        if len(losses) >= 2:
            print(f"egitim kaybi: ilk {losses[0]:.3f} -> son {losses[-1]:.3f}")
            if losses[-1] > 0.5 * losses[0]:
                print("UYARI: kayip yeterince dusmedi. Gercek veride ayni testi tekrarlayin;")
                print("        orada da dusmuyorsa mimaride veya veri hazirliginda sorun var.")

    ok, _ = run("7. karsilastirma tablosu", [PY, f"{s}/run_seeds.py", "--table", "--out_dir", OUT],
                must_exist=[f"{OUT}/comparison.md"])

    print("\n=========== OZET ===========")
    for name, ok, dt in steps:
        print(f"[{' OK ' if ok else 'HATA'}] {name}  ({dt:.1f} sn)")
    sys.exit(0 if all(ok for _, ok, _ in steps) else 1)


if __name__ == "__main__":
    main()
