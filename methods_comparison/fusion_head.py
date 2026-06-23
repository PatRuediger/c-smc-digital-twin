import numpy as np
import torch
from torch import nn

class FusionHead(nn.Module):
    def __init__(self, feature_dim: int, topo_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + topo_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, feat: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feat, topo], dim=1))

def train_fusion(
    X_train: np.ndarray, T_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, T_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 50, lr: float = 1e-3, patience: int = 10,
) -> tuple[FusionHead, dict]:
    head = FusionHead(feature_dim=X_train.shape[1], topo_dim=T_train.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_train); Tt = torch.from_numpy(T_train); yt = torch.from_numpy(y_train).unsqueeze(1)
    Xv = torch.from_numpy(X_val);   Tv = torch.from_numpy(T_val);   yv = torch.from_numpy(y_val).unsqueeze(1)
    log = {"train_loss": [], "val_mae": []}
    best, since_best = float("inf"), 0
    best_state = None
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        pred = head(Xt, Tt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            mae = float((head(Xv, Tv) - yv).abs().mean())
        log["train_loss"].append(float(loss.detach()))
        log["val_mae"].append(mae)
        if mae < best:
            best, since_best = mae, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, log
