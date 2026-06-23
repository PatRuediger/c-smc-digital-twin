from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

from methods_comparison.bootstrap_ci import mae_ci, paired_mae_diff_ci

METHOD_COLS = ("watershed_pred", "csrnet_pred", "ttk_pred", "fusion_pred")
REGIMES = ("low", "medium", "high")
DISPLAY_LABEL = {
    "watershed": "Watershed",
    "csrnet": "CSRNet",
    "ttk": "MSC",
    "fusion": "Fusion",
}


def main(results: Path, out_dir: Path) -> None:
    df = pd.read_csv(results)
    methods = [m for m in METHOD_COLS if m in df.columns]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in methods:
        for regime in REGIMES:
            sub = df[df["density_regime"] == regime]
            if len(sub) == 0:
                continue
            err = np.asarray(sub[m] - sub["gt_count"])
            mae, lo, hi = mae_ci(err)
            rmse = float(np.sqrt((err ** 2).mean()))
            rows.append({
                "method": m.replace("_pred", ""),
                "regime": regime,
                "mae": mae,
                "mae_ci_lo": lo,
                "mae_ci_hi": hi,
                "rmse": rmse,
                "n": int(len(sub)),
            })
    mdf = pd.DataFrame(rows)
    mdf.to_csv(out_dir / "table1_metrics.csv", index=False)

    pairs = []
    for ra, rb in [("watershed_pred", "csrnet_pred"),
                   ("watershed_pred", "ttk_pred"),
                   ("csrnet_pred", "ttk_pred"),
                   ("csrnet_pred", "fusion_pred")]:
        if ra not in df.columns or rb not in df.columns:
            continue
        ea = np.asarray(df[ra] - df["gt_count"])
        eb = np.asarray(df[rb] - df["gt_count"])
        diff, lo, hi = paired_mae_diff_ci(ea, eb)
        pairs.append({
            "pair": f"{ra.replace('_pred', '')} - {rb.replace('_pred', '')}",
            "mae_diff": diff,
            "ci_lo": lo,
            "ci_hi": hi,
            "favors_b": diff > 0 and lo > 0,
            "favors_a": diff < 0 and hi < 0,
        })
    pd.DataFrame(pairs).to_csv(out_dir / "table2_paired_diffs.csv", index=False)

    # Per-regime GT statistics for x-tick annotations (orientation reference)
    regime_gt = {
        r: df[df["density_regime"] == r]["gt_count"]
        for r in REGIMES
    }
    regime_label = {
        r: f"{r}\n(GT range\n{int(g.min())}–{int(g.max())},\nmean {int(g.mean())})"
        for r, g in regime_gt.items()
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.8 / len(methods)
    x = np.arange(len(REGIMES))
    for i, m in enumerate(methods):
        key = m.replace("_pred", "")
        sub = mdf[mdf["method"] == key].set_index("regime").reindex(REGIMES)
        ax.bar(x + i * width, sub["mae"], width=width,
               yerr=[sub["mae"] - sub["mae_ci_lo"], sub["mae_ci_hi"] - sub["mae"]],
               capsize=3, label=DISPLAY_LABEL.get(key, key))
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([regime_label[r] for r in REGIMES])
    ax.set_ylabel("MAE [strips] (95% CI)")
    ax.set_xlabel("Density regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_quantitative.pdf", dpi=300)
    plt.close(fig)

    n = len(methods)
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3.2), sharex=True, sharey=True)
    if n == 1:
        axs = [axs]
    for ax, m in zip(axs, methods):
        ax.scatter(df["gt_count"], df[m], s=10)
        lo, hi = df["gt_count"].min(), df["gt_count"].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        key = m.replace("_pred", "")
        ax.set_title(DISPLAY_LABEL.get(key, key))
        ax.set_xlabel("GT count")
    axs[0].set_ylabel("Predicted count")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_scatter.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main(Path("methods_comparison/results.csv"),
         Path(os.environ.get("FIGURE_OUTPUT", "./figures")))
