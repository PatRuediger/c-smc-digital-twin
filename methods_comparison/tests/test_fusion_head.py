import numpy as np
import torch

from methods_comparison._constants import CSRNET_FEATURE_DIM
from methods_comparison.fusion_head import FusionHead, train_fusion

def test_fusion_head_forward_shape():
    head = FusionHead(feature_dim=CSRNET_FEATURE_DIM, topo_dim=6)
    x_feat = torch.zeros(4, CSRNET_FEATURE_DIM)
    x_topo = torch.zeros(4, 6)
    out = head(x_feat, x_topo)
    assert out.shape == (4, 1)

def test_train_fusion_runs_and_drops_loss():
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, CSRNET_FEATURE_DIM)).astype(np.float32)
    T = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X.sum(axis=1) + T.sum(axis=1)).astype(np.float32)
    _, log = train_fusion(X[:48], T[:48], y[:48], X[48:], T[48:], y[48:], epochs=20)
    assert log["val_mae"][-1] < log["val_mae"][0]
