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

import os

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


def _first_test_frame() -> Path:
    """First test frame from splits.json, resolved against the Exp_09 data root."""
    import json
    entry = json.loads(Path("methods_comparison/splits.json").read_text())["test"][0]
    path = Path(entry["image_path"])
    if not path.is_absolute():
        root = Path(os.environ.get("EXP09_ROOT", "./data/Exp_09")).parent
        path = root / path
    return path


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not yet trained")
@pytest.mark.parametrize("preproc,should_match", [("train", True), ("legacy", False)])
def test_preprocess_matches_training_pipeline(preproc, should_match):
    """Inference input must equal what csrnet/train_csrnet.py fed the net.

    Regression for the 2026-11-27 finding: the legacy path (grayscale, no
    ImageNet Normalize) compressed the count predictions to ~300-380.
    """
    from PIL import Image
    from torchvision import transforms

    frame = _first_test_frame()
    if not frame.exists():
        pytest.skip(f"test frame not found: {frame} (set EXP09_ROOT)")
    # Training pipeline: csrnet_dataset.py (convert RGB) + train_csrnet.py transform.
    train_tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    expected = train_tx(Image.open(frame).convert("RGB")).unsqueeze(0)
    got = CSRNetInfer(CKPT, device="cpu", preproc=preproc).preprocess(frame).cpu()
    assert got.shape == expected.shape
    assert torch.allclose(got, expected) is should_match
