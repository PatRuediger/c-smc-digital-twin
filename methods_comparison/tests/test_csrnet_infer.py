"""Tests for CSRNetInfer inference wrapper.

Must be run with a Python interpreter that has torch installed.
Install pytest into the venv first (one-time):
    ivw_dt_pytorch/bin/pip install pytest

Then run with:
    python3 \
        -m pytest methods_comparison/tests/test_csrnet_infer.py -v

The test is skipped until the checkpoint exists (training is a human-driven step).
After training completes, re-run the test to verify GREEN.
"""

import numpy as np
from pathlib import Path
import pytest

torch = pytest.importorskip("torch", reason="torch not installed in this interpreter")

from methods_comparison.csrnet_infer import CSRNetInfer  # noqa: E402
from methods_comparison._constants import CSRNET_FEATURE_DIM  # noqa: E402

CKPT = Path("methods_comparison/checkpoints/csrnet_best.pth")


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not yet trained")
def test_predict_count_returns_int_and_feature_vector():
    inf = CSRNetInfer(CKPT)
    test_img = next(
        Path(
            os.environ.get("EXP09_ROOT", "./data/Exp_09")
        ).rglob("*.png")
    )
    count, feat = inf.predict_count(test_img)
    assert isinstance(count, int)
    assert count >= 0
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (CSRNET_FEATURE_DIM,)
