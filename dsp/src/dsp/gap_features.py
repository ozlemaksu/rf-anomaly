import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from analyze_band_ratio_v2 import (
    L, WPF, estimate_bw, find_iq, meta_bw, read_iq, window_power,
)


def runs(mask):
    """True değerlerin ardışık bloklarını [(başlangıç, uzunluk)] olarak döndürür."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)

    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]

    return list(zip(starts, ends - starts))


def safe_spearman(x, y):
    """
    Spearman korelasyonunu güvenli şekilde hesaplar.
    Veri sabitse veya yeterli veri yoksa NaN döndürür.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return spearmanr(x, y).statistic


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--iq-dir", required=True)
    ap.add_argument("--meta-dir", default=None)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="gap_out")

    ap.add_argument("--nfft", type=int, default=256)
    ap.add_argument("--hop", type=int, default=256)

    ap.add_argument("--bw-col", default=None)

    ap.add_argument(
        "--min-gap-ms",
        type=float,
        default=0.5,
        help="Uzun gap olarak kabul edilecek minimum süre (ms).",
    )

    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # ---------------------------------------------------------
    # Prediction CSV
    # ---------------------------------------------------------

    df = pd.read_csv(a.pred)

    required_columns = [
        "file",
        "window_in_file",
        "fs_hz",
        "snr_db",
        "sir_db",
        "idx",
        "y_true",
        "cnn2d_prob",
        "cnn2d_error",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Prediction CSV içinde eksik sütunlar var: {missing_columns}"
        )

    rows = []
    missing_recordings = 0

    # ---------------------------------------------------------
    # Recording bazında işle
    # ---------------------------------------------------------

    for name, g in df.groupby("file"):

        path = find_iq(a.iq_dir, name)

        if path is None:
            missing_recordings += 1
            continue

        i, q = read_iq(path)

        fs = float(g.fs_hz.iloc[0])

        nominal, src = meta_bw(
            a.meta_dir,
            name,
            a.bw_col,
        )

        if nominal is not None:
            bw = 0.9 * nominal
        else:
            bw = estimate_bw(i, q, fs)
            src = "sinyal"

        # 4 adet CNN penceresi
        if len(i) < L:
            print(
                f"[UYARI] {name}: "
                f"IQ uzunluğu ({len(i)}) L={L}'den küçük."
            )
            continue

        starts = np.linspace(
            0,
            len(i) - L,
            WPF,
        ).astype(int)

        # STFT zaman kolonunun süresi
        col_ms = a.hop / fs * 1e3

        cin = {}
        cout = {}
        ratio = {}

        # -----------------------------------------------------
        # Her recording için 4 pencerenin STFT'i
        # -----------------------------------------------------

        for w in range(WPF):

            f, P = window_power(
                i,
                q,
                fs,
                starts[w],
                a.nfft,
                a.hop,
            )

            # LTE bandı
            inb = np.abs(f) <= 0.8 * bw / 2

            # LTE bandı dışı
            oob = np.abs(f) >= 1.2 * bw / 2

            if not np.any(inb):
                raise ValueError(
                    f"{name}: in-band frekans binleri bulunamadı."
                )

            if not np.any(oob):
                raise ValueError(
                    f"{name}: out-of-band frekans binleri bulunamadı."
                )

            # Her zaman kolonu için ortalama in-band / out-of-band güç
            cin[w] = 10 * np.log10(
                np.maximum(
                    P[inb].mean(axis=0),
                    1e-30,
                )
            )

            cout[w] = 10 * np.log10(
                np.maximum(
                    P[oob].mean(axis=0),
                    1e-30,
                )
            )

            # Recording genel band ratio
            ratio[w] = 10 * np.log10(
                max(P[inb].mean(), 1e-30)
                /
                max(P[oob].mean(), 1e-30)
            )

        # -----------------------------------------------------
        # Recording seviyesinde noise / on-level
        # -----------------------------------------------------

        all_cout = np.concatenate(list(cout.values()))
        all_cin = np.concatenate(list(cin.values()))

        noise = float(np.median(all_cout))

        on = float(
            np.percentile(
                all_cin,
                90,
            )
        )

        contrast = on - noise

        # Gap threshold
        thr = noise + 0.5 * contrast

        # -----------------------------------------------------
        # Prediction CSV'deki her window
        # -----------------------------------------------------

        for _, r in g.iterrows():

            w = int(r.window_in_file)

            if w not in cin:
                continue

            # Bu window'daki zaman kolonları
            # threshold altındaysa gap
            gap = cin[w] < thr

            # Ardışık gap blokları
            rr = runs(gap)

            # Minimum süreyi geçen gap'ler
            long_ = [
                (s, n)
                for s, n in rr
                if n * col_ms >= a.min_gap_ms
            ]

            # En uzun gap
            max_gap = max(
                [n for _, n in rr],
                default=0,
            )

            rows.append(
                dict(
                    idx=int(r.idx),
                    file=name,
                    window_in_file=w,
                    fs_hz=float(r.fs_hz),
                    snr_db=float(r.snr_db),
                    sir_db=(
                        float(r.sir_db)
                        if pd.notna(r.sir_db)
                        else np.nan
                    ),
                    y_true=int(r.y_true),
                    cnn2d_prob=float(r.cnn2d_prob),
                    cnn2d_error=r.cnn2d_error,

                    # Daha önce hesapladığımız özellik
                    band_ratio_db=round(
                        float(ratio[w]),
                        3,
                    ),

                    # Yeni gap özellikleri
                    gap_frac=round(
                        float(gap.mean()),
                        3,
                    ),

                    long_gap_frac=round(
                        float(
                            sum(
                                n for _, n in long_
                            )
                            /
                            len(gap)
                        ),
                        3,
                    ),

                    max_gap_ms=round(
                        float(
                            max_gap * col_ms
                        ),
                        3,
                    ),

                    n_long_gaps=len(long_),

                    # Recording'in genel enerji kontrastı
                    contrast_db=round(
                        float(contrast),
                        2,
                    ),

                    # Contrast < 1.5 dB ise feature güvenilmez
                    guvenilir=bool(
                        contrast >= 1.5
                    ),
                )
            )

    # ---------------------------------------------------------
    # Sonuçları DataFrame'e dönüştür
    # ---------------------------------------------------------

    if missing_recordings:
        print(
            f"[UYARI] IQ bulunamayan recording: "
            f"{missing_recordings}"
        )

    if not rows:
        raise RuntimeError(
            "Hiçbir pencere işlenemedi."
        )

    d = (
        pd.DataFrame(rows)
        .sort_values("idx")
        .reset_index(drop=True)
    )

    output_csv = os.path.join(
        a.out,
        "gap_features_table.csv",
    )

    d.to_csv(
        output_csv,
        index=False,
    )

    print()
    print(
        f"{d.file.nunique()} recording, "
        f"{len(d)} pencere / {len(df)}"
    )

    print(
        "Güvenilmez feature "
        f"(contrast < 1.5 dB): "
        f"{(~d.guvenilir).sum()}"
    )

    # ---------------------------------------------------------
    # Durum
    # ---------------------------------------------------------

    d["durum"] = np.where(
        d.cnn2d_error == "dogru",
        "dogru",
        d.cnn2d_error,
    )

    # ---------------------------------------------------------
    # Grup özeti
    # ---------------------------------------------------------

    pd.set_option(
        "display.width",
        220,
    )

    print()
    print(
        "Grup özeti "
        "(sinif, SNR, SIR, durum):"
    )

    group_summary = (
        d.groupby(
            [
                "y_true",
                "snr_db",
                "sir_db",
                "durum",
            ],
            dropna=False,
        )
        .agg(
            n=("idx", "size"),
            gap_frac=("gap_frac", "mean"),
            long_gap=("long_gap_frac", "mean"),
            max_gap_ms=("max_gap_ms", "mean"),
            band_ratio=("band_ratio_db", "mean"),
            contrast=("contrast_db", "mean"),
            p2d=("cnn2d_prob", "mean"),
        )
        .round(3)
    )

    print(
        group_summary.to_string()
    )

    # ---------------------------------------------------------
    # Spearman
    # ---------------------------------------------------------

    features = [
        "gap_frac",
        "long_gap_frac",
        "max_gap_ms",
        "band_ratio_db",
    ]

    print()
    print(
        "Spearman cnn2d_prob ~ özellik "
        "(n>=8 gruplar):"
    )

    for (y_, sn), g in d.groupby(
        ["y_true", "snr_db"]
    ):

        if len(g) < 8:
            continue

        results = []

        for feature in features:

            rho = safe_spearman(
                g[feature],
                g["cnn2d_prob"],
            )

            if np.isnan(rho):
                results.append(
                    f"{feature}=nan"
                )
            else:
                results.append(
                    f"{feature}={rho:.2f}"
                )

        print(
            f"  y={y_} "
            f"SNR={sn:g} "
            f"n={len(g)}: "
            + "  ".join(results)
        )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    col = {
        "dogru": "0.65",
        "FP": "tab:red",
        "FN": "tab:blue",
    }

    mk = {
        7.68e6: "o",
        15.36e6: "s",
        30.72e6: "^",
    }

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        sharey=True,
    )

    plot_features = [
        "gap_frac",
        "long_gap_frac",
        "max_gap_ms",
    ]

    plot_labels = [
        "gap_frac (tüm boşluklar)",
        f"long_gap_frac (>={a.min_gap_ms} ms)",
        "max_gap_ms",
    ]

    for ax, feature, label in zip(
        axs,
        plot_features,
        plot_labels,
    ):

        for status in col:

            for fs_, marker in mk.items():

                s = d[
                    (d.durum == status)
                    &
                    (d.fs_hz == fs_)
                ]

                if len(s) == 0:
                    continue

                ax.scatter(
                    s[feature],
                    s.cnn2d_prob,
                    c=col[status],
                    marker=marker,
                    s=(
                        45
                        if status == "dogru"
                        else 75
                    ),
                    edgecolors=(
                        "k"
                        if status != "dogru"
                        else "none"
                    ),
                    linewidths=0.6,
                    label=(
                        f"{status} "
                        f"{fs_ / 1e6:g} MHz"
                    ),
                )

        ax.axhline(
            0.5,
            color="k",
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_xlabel(label)
        ax.grid(alpha=0.3)

    axs[0].set_ylabel(
        "cnn2d_prob"
    )

    axs[0].legend(
        fontsize=6,
        ncol=2,
    )

    fig.tight_layout()

    plot_path = os.path.join(
        a.out,
        "gap_vs_prob.png",
    )

    fig.savefig(
        plot_path,
        dpi=130,
    )

    plt.close(fig)

    print()
    print(f"CSV: {output_csv}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()