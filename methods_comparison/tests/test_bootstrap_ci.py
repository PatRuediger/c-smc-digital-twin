import numpy as np
from methods_comparison.bootstrap_ci import mae_ci, paired_mae_diff_ci


def test_mae_ci_brackets_mean():
    err = np.array([1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.0, -1.0])
    mean, lo, hi = mae_ci(err, n_boot=500, seed=42)
    assert abs(mean - np.abs(err).mean()) < 1e-9
    assert lo <= mean <= hi


def test_paired_diff_ci_zero_when_identical():
    err = np.array([1.0, -2.0, 3.0, -4.0])
    mean, lo, hi = paired_mae_diff_ci(err, err, n_boot=500, seed=42)
    assert abs(mean) < 1e-9
    assert lo <= 0 <= hi
