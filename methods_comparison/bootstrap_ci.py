"""Paired Bootstrap-CIs for MAE and MAE-Differenzen."""
import numpy as np


def mae_ci(err: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    abs_err = np.abs(err)
    boots = np.array([abs_err[rng.integers(0, len(abs_err), len(abs_err))].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(abs_err.mean()), float(lo), float(hi)


def paired_mae_diff_ci(err_a: np.ndarray, err_b: np.ndarray,
                       n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Diff = MAE_A - MAE_B. Negative -> A besser."""
    rng = np.random.default_rng(seed)
    n = len(err_a)
    assert n == len(err_b)
    a, b = np.abs(err_a), np.abs(err_b)
    boots = np.array([
        (a[(idx := rng.integers(0, n, n))].mean() - b[idx].mean())
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)
