"""Plottet Histogramme von 20 AoLP-Sample-Bildern und berechnet Bimodalitaets-Index.

PASS: median Bimodalitaets-Index >= 0.15 (deutliche Tal-Tiefe) -> Otsu+Watershed plausibel.
WARN: 0.05 <= Index < 0.15 -> User-Decision: weiter mit Watershed oder auf gradient-of-AoLP umstellen.
FAIL: Index < 0.05 -> Plan Task 2 muss umgebaut werden auf gradient-magnitude-of-AoLP oder DoLP.

Note: image paths in the DB are relative to EXP09_ROOT.
Note: skimage is not available; threshold_otsu is implemented inline (Otsu 1979).
"""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

EXP09_ROOT = Path(os.environ.get("EXP09_ROOT", "./data/Exp_09"))
PYTORCH_VENV_PYTHON = Path(
    "python3"
)


def _reexec_if_needed() -> None:
    """Re-exec with the pytorch venv if numpy/PIL/matplotlib are missing."""
    from importlib.util import find_spec
    required = ("numpy", "PIL", "matplotlib")
    missing = [m for m in required if find_spec(m) is None]
    if not missing:
        return
    if os.environ.get("PREFLIGHT_REEXEC") == "1":
        print(f"FAIL: required packages still missing after re-exec: {missing}")
        sys.exit(1)
    env = dict(os.environ)
    env["PREFLIGHT_REEXEC"] = "1"
    result = subprocess.run(
        [str(PYTORCH_VENV_PYTHON), "-m", "methods_comparison.preflight.check_aolp_histograms"],
        env=env,
    )
    sys.exit(result.returncode)


_reexec_if_needed()


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # Read constants after ensuring we are in the right venv
    sys.path.insert(0, str(Path(".").resolve()))
    from methods_comparison._constants import EXP09_TABLE, EXP09_COL_IMAGE

    def threshold_otsu_inline(arr: np.ndarray) -> float:
        """Compute Otsu threshold (maximize inter-class variance). Equivalent to skimage.filters.threshold_otsu."""
        counts, bin_edges = np.histogram(arr.flatten(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = counts.sum()
        if total == 0:
            return float(bin_centers[128])
        weight = counts / total
        # Cumulative sums for class weights and means
        w0 = np.cumsum(weight)
        w1 = 1.0 - w0
        mean0 = np.cumsum(weight * bin_centers) / np.where(w0 > 0, w0, 1)
        mean_total = (weight * bin_centers).sum()
        mean1 = np.where(w1 > 0, (mean_total - np.cumsum(weight * bin_centers)) / np.where(w1 > 0, w1, 1), 0.0)
        variance = w0 * w1 * (mean0 - mean1) ** 2
        return float(bin_centers[np.argmax(variance)])

    def bimodality_index(arr: np.ndarray, bins: int = 64) -> float:
        """Valley-depth metric: relative depth of the deepest valley between the two highest peaks."""
        hist, _ = np.histogram(arr.flatten(), bins=bins)
        if hist.max() == 0:
            return 0.0
        smoothed = np.convolve(hist, np.ones(3) / 3, mode="same")
        # find two dominant peaks
        order = np.argsort(smoothed)[::-1]
        top1 = order[0]
        top2 = next((i for i in order[1:] if abs(i - top1) > bins // 8), order[1])
        lo, hi = sorted((top1, top2))
        valley = smoothed[lo:hi].min() if hi > lo else smoothed[top1]
        peak = min(smoothed[top1], smoothed[top2])
        return float((peak - valley) / peak) if peak > 0 else 0.0

    dbs = sorted(EXP09_ROOT.glob("Exp_09_Polarized_OBB_*/stripGen_results.db"))[:20]
    samples: list[tuple[str, np.ndarray]] = []
    for db in dbs:
        with sqlite3.connect(db) as conn:
            row = conn.execute(f"SELECT {EXP09_COL_IMAGE} FROM {EXP09_TABLE} LIMIT 1").fetchone()
            if not row:
                print(f"  WARN: no rows in {db.parent.name}")
                continue
            img_rel = row[0]
            img_path = EXP09_ROOT / img_rel
            if not img_path.exists():
                print(f"  WARN: image not found: {img_path}")
                continue
            arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
            samples.append((db.parent.name, arr))

    if not samples:
        print("FAIL: no AoLP images found; check EXP09_ROOT and DB paths")
        return 1

    indices = [bimodality_index(a) for _, a in samples]
    otsu_thresholds = [threshold_otsu_inline(a) for _, a in samples]
    median_bi = float(np.median(indices))

    n_cols = 5
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for _, (ax, (name, arr), bi, otsu) in enumerate(
        zip(axes_flat, samples, indices, otsu_thresholds)
    ):
        ax.hist(arr.flatten(), bins=64, color="steelblue")
        ax.axvline(otsu, color="red", lw=1, label=f"Otsu={otsu:.0f}")
        ax.set_title(f"{name[-15:]}\nBI={bi:.2f}", fontsize=7)
        ax.legend(fontsize=6)
    # hide unused axes
    for ax in axes_flat[len(samples):]:
        ax.set_visible(False)
    fig.tight_layout()

    preflight_dir = Path("methods_comparison/preflight")
    preflight_dir.mkdir(parents=True, exist_ok=True)
    out = preflight_dir / "aolp_histograms.pdf"
    fig.savefig(out)
    plt.close(fig)

    if median_bi >= 0.15:
        verdict = "PASS"
        ret = 0
    elif median_bi >= 0.05:
        verdict = "WARN"
        ret = 0
    else:
        verdict = "FAIL"
        ret = 1
    print(f"{verdict}: median Bimodality Index = {median_bi:.3f} on n={len(samples)} AoLP samples")
    print(f"histograms: {out}")
    if verdict == "WARN":
        print("WARN: histograms borderline. Inspect aolp_histograms.pdf and decide:")
        print("  (a) keep Otsu+Watershed -> proceed")
        print("  (b) switch Task 2 input to gradient-magnitude(AoLP) or DoLP image")
    if verdict == "FAIL":
        print("FAIL: Otsu/Watershed on raw AoLP is unlikely to work.")
        print("Plan-Update: Task 2 erhaelt einen Pre-Processing-Step (Sobel auf AoLP oder DoLP-Kanal).")
    return ret


if __name__ == "__main__":
    sys.exit(main())
