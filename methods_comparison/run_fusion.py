"""Trainiert FusionHead, ergaenzt fusion_pred-Spalte in results.csv.

Features and targets are z-score normalised before training (fit on train),
because raw CSRNet features (~[0, 10]) and raw TTK summary (n_pairs ~1000,
max_persistence ~50, betti ~100s) live on wildly different scales, and the
target gt_count ranges 0-700. Without normalisation the MSE-loss starts at
~1e10 and the MLP cannot converge.
"""
import csv
import json
from pathlib import Path

import numpy as np

from methods_comparison.csrnet_infer import CSRNetInfer
from methods_comparison.data import load_split
from methods_comparison.fusion_head import train_fusion

TOPO_KEYS = ("n_pairs", "max_persistence", "total_persistence",
             "persistence_entropy", "betti_dim0", "betti_dim1")


def _build_xy(split: str, csrnet: CSRNetInfer, topo_cache: Path):
    X, T, y = [], [], []
    for e in load_split(Path("methods_comparison/splits.json"), split):
        _, feat = csrnet.predict_count(Path(e.image_path))
        topo = json.load(open(topo_cache / f"{e.run_id}_{e.frame_id}.json"))["persistence_summary"]
        X.append(feat)
        T.append([float(topo[k]) for k in TOPO_KEYS])
        y.append(float(e.gt_count))
    return (np.array(X, dtype=np.float32),
            np.array(T, dtype=np.float32),
            np.array(y, dtype=np.float32))


def _fit_zscore(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return arr.mean(axis=0), arr.std(axis=0) + 1e-6


def _apply_zscore(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr - mean) / std).astype(np.float32)


def main():
    csrnet = CSRNetInfer(Path("methods_comparison/checkpoints/csrnet_best.pth"))
    topo_cache = Path("methods_comparison/cache/topology")

    print("Building (CSRNet feature, TTK summary) tensors per split...")
    Xt_raw, Tt_raw, yt_raw = _build_xy("train", csrnet, topo_cache)
    Xv_raw, Tv_raw, yv_raw = _build_xy("val",   csrnet, topo_cache)
    Xs_raw, Ts_raw, _      = _build_xy("test",  csrnet, topo_cache)
    print(f"  train: X={Xt_raw.shape} T={Tt_raw.shape} y range [{yt_raw.min():.0f}, {yt_raw.max():.0f}]")

    # Fit normalisation stats on train, apply to all splits.
    X_mean, X_std = _fit_zscore(Xt_raw)
    T_mean, T_std = _fit_zscore(Tt_raw)
    y_max = max(float(yt_raw.max()), 1.0)

    Xt = _apply_zscore(Xt_raw, X_mean, X_std)
    Tt = _apply_zscore(Tt_raw, T_mean, T_std)
    Xv = _apply_zscore(Xv_raw, X_mean, X_std)
    Tv = _apply_zscore(Tv_raw, T_mean, T_std)
    Xs = _apply_zscore(Xs_raw, X_mean, X_std)
    Ts = _apply_zscore(Ts_raw, T_mean, T_std)
    yt_n = (yt_raw / y_max).astype(np.float32)
    yv_n = (yv_raw / y_max).astype(np.float32)

    print(f"  normalised: y_max={y_max}, training MLP...")
    head, log = train_fusion(Xt, Tt, yt_n, Xv, Tv, yv_n, epochs=200, patience=20)

    # Persist normalisation params + log so the fusion is reproducible.
    norm = {
        "X_mean": X_mean.tolist(), "X_std": X_std.tolist(),
        "T_mean": T_mean.tolist(), "T_std": T_std.tolist(),
        "y_max": y_max,
    }
    Path("methods_comparison/fusion_norm.json").write_text(json.dumps(norm))
    Path("methods_comparison/fusion_log.json").write_text(json.dumps(log))
    best_val_mae_normed = min(log["val_mae"])
    print(f"  best val MAE (normalised space): {best_val_mae_normed:.4f}")
    print(f"  best val MAE (counts):           {best_val_mae_normed * y_max:.2f}")

    import torch
    head.eval()
    with torch.no_grad():
        pred_normed = head(torch.from_numpy(Xs), torch.from_numpy(Ts)).squeeze().numpy()
    pred = pred_normed * y_max  # de-normalise

    rows = list(csv.DictReader(open("methods_comparison/results.csv")))
    assert len(rows) == len(pred), f"results.csv has {len(rows)} rows but fusion produced {len(pred)} preds"
    for row, p in zip(rows, pred):
        row["fusion_pred"] = max(0, int(round(float(p))))  # clip negatives
    fieldnames = list(rows[0].keys())
    if "fusion_pred" not in fieldnames:
        fieldnames.append("fusion_pred")
    with open("methods_comparison/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote fusion_pred for {len(rows)} test rows.")


if __name__ == "__main__":
    main()
