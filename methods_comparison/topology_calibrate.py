"""Fit linear calibration gt = a * raw + b on validation data.

Reads raw TTK MSC counts from JSON cache and ground-truth counts from splits.json.
Fits a linear regression model and persists coefficients to topology_calibration.json.

The calibration accounts for the ~18x scale difference between raw ascending-manifold
counts and ground-truth strip counts, observed during TTK preflight testing.

Usage:
    python3 -m methods_comparison.topology_calibrate

Inputs:
    methods_comparison/splits.json (val split)
    methods_comparison/cache/topology/{run_id}_{frame_id}.json (raw TTK counts)

Output:
    methods_comparison/topology_calibration.json with {"a": float, "b": float, "n_val": int, "r_pearson": float}

If <5 cached val frames: skip with a clear message; the user must run val TTK first.
"""

import json
from pathlib import Path

import numpy as np

from methods_comparison.data import load_split


def fit_calibration() -> dict | None:
    """Fit linear regression on validation data.

    Returns
    -------
    dict | None
        Calibration coefficients {"a", "b", "n_val", "r_pearson"} if successful.
        None if insufficient val cache.
    """
    splits = Path("methods_comparison/splits.json")
    cache = Path("methods_comparison/cache/topology")

    raw, gt = [], []
    for entry in load_split(splits, "val"):
        cache_file = cache / f"{entry.run_id}_{entry.frame_id}.json"
        if cache_file.exists():
            payload = json.loads(cache_file.read_text())
            raw.append(float(payload["count"]))
            gt.append(float(entry.gt_count))

    if len(raw) < 5:
        print(f"only {len(raw)} val cache files; need >= 5 to fit calibration")
        return None

    raw_a = np.array(raw, dtype=float)
    gt_a = np.array(gt, dtype=float)

    # Linear fit: gt = a * raw + b
    a, b = np.polyfit(raw_a, gt_a, 1)

    # Pearson correlation
    r = float(np.corrcoef(raw_a, gt_a)[0, 1])

    coeffs = {
        "a": float(a),
        "b": float(b),
        "n_val": len(raw),
        "r_pearson": r,
    }

    Path("methods_comparison/topology_calibration.json").write_text(
        json.dumps(coeffs, indent=2)
    )
    print(f"PASS: a={a:.5f}, b={b:.2f}, n_val={len(raw)}, r={r:.3f}")
    return coeffs


if __name__ == "__main__":
    fit_calibration()
