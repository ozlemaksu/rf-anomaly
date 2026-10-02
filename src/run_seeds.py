"""Bir modeli birden fazla seed ile calistirir, ortalama +- std uretir; karsilastirma tablosunu yazar.

Calistirma:
    python src/run_seeds.py --model cnn1d --x data/X_iq.npy   --y data/y.npy --splits data/splits.npz
    python src/run_seeds.py --model cnn2d --x data/X_spec.npy --y data/y.npy --splits data/splits.npz
    python src/baseline_energy.py --x data/X_iq.npy --y data/y.npy --splits data/splits.npz
    python src/run_seeds.py --table            # results/comparison.md (sunumdaki tablo)

Her seed ayri dosyaya yazilir (<model>_s<seed>_metrics.json), birbirinin uzerine yazmaz.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ROWS = [  # (dosya oneki, tabloda gorunen ad, girdi)
    ("baseline_energy", "Baseline (enerji esigi)", "Ham IQ gucu"),
    ("cnn1d", "1D CNN", "Ham IQ (I, Q)"),
    ("cnn2d", "2D CNN", "STFT spektrogrami"),
]
KEYS = ["acc", "f1", "auc", "latency_ms"]


def aggregate(out_dir, model, seeds):
    """<model>_s<seed>_metrics.json dosyalarindan <model>_summary.json uretir."""
    runs = []
    for s in seeds:
        path = os.path.join(out_dir, f"{model}_s{s}_metrics.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} yok: seed {s} calismamis olabilir")
        with open(path) as f:
            runs.append(json.load(f))
    summary = {"model": model, "seeds": list(seeds), "n_runs": len(runs),
               "n_params": runs[0]["n_params"]}
    for k in KEYS:
        v = np.array([r[k] for r in runs], dtype=float)
        summary[k + "_mean"] = float(np.nanmean(v))
        summary[k + "_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
    with open(os.path.join(out_dir, f"{model}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def fmt(mean, std, nd=3):
    return f"{mean:.{nd}f} ± {std:.{nd}f}" if std > 1e-9 else f"{mean:.{nd}f}"


def build_table(out_dir):
    lines = [
        "| Model | Girdi | Doğruluk | F1 | ROC-AUC | Çıkarım (ms/örnek) | Parametre | Seed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for prefix, name, inp in ROWS:
        summ = os.path.join(out_dir, f"{prefix}_summary.json")
        single = os.path.join(out_dir, f"{prefix}_metrics.json")
        if os.path.exists(summ):
            with open(summ) as f:
                m = json.load(f)
            lines.append(
                f"| {name} | {inp} | {fmt(m['acc_mean'], m['acc_std'])} | {fmt(m['f1_mean'], m['f1_std'])} "
                f"| {fmt(m['auc_mean'], m['auc_std'])} | {fmt(m['latency_ms_mean'], m['latency_ms_std'], 2)} "
                f"| {m['n_params']:,} | {m['n_runs']} |"
            )
        elif os.path.exists(single):
            with open(single) as f:
                m = json.load(f)
            lines.append(
                f"| {name} | {inp} | {m['acc']:.3f} | {m['f1']:.3f} | {m['auc']:.3f} "
                f"| {m['latency_ms']:.2f} | {m['n_params']:,} | 1 |"
            )
        else:
            lines.append(f"| {name} | {inp} | | | | | | |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows: Turkce karakterler ve ± icin
    p = argparse.ArgumentParser()
    p.add_argument("--table", action="store_true", help="yalnizca karsilastirma tablosunu uret")
    p.add_argument("--model", choices=["cnn1d", "cnn2d"])
    p.add_argument("--x")
    p.add_argument("--y")
    p.add_argument("--splits")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--out_dir", default="results")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    if a.table:
        print(build_table(a.out_dir))
        return
    if not (a.model and a.x and a.y and a.splits):
        p.error("--model, --x, --y, --splits gerekli (ya da yalniz --table)")

    for s in a.seeds:
        print(f"\n=== {a.model}  seed {s} ===", flush=True)
        cmd = [
            sys.executable, os.path.join(HERE, "train.py"),
            "--model", a.model, "--x", a.x, "--y", a.y, "--splits", a.splits,
            "--epochs", str(a.epochs), "--bs", str(a.bs), "--lr", str(a.lr),
            "--patience", str(a.patience), "--seed", str(s),
            "--out_dir", a.out_dir, "--tag", f"{a.model}_s{s}",
        ]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"seed {s} basarisiz (cikis kodu {r.returncode})")

    summ = aggregate(a.out_dir, a.model, a.seeds)
    print(f"\n{a.model} ({summ['n_runs']} seed): "
          f"acc {fmt(summ['acc_mean'], summ['acc_std'])}  "
          f"F1 {fmt(summ['f1_mean'], summ['f1_std'])}  "
          f"AUC {fmt(summ['auc_mean'], summ['auc_std'])}")
    print(build_table(a.out_dir))


if __name__ == "__main__":
    main()
