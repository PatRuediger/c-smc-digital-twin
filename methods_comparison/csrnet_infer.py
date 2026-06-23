"""CSRNet inference wrapper for strip-count prediction and feature extraction.

Usage:
    Run with the pytorch venv, not system python3:
        python3

    Example:
        from pathlib import Path
        from methods_comparison.csrnet_infer import CSRNetInfer

        inf = CSRNetInfer(Path("methods_comparison/checkpoints/csrnet_best.pth"))
        count, feat = inf.predict_count(Path("/path/to/image.png"))
        # count: int  (rounded density-map sum)
        # feat:  np.ndarray shape (512,)  (Global-Average-Pooled feature from backend.4)
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ensure project root is on sys.path so 'csrnet' resolves as a package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from csrnet.csrnet_model import CSRNet  # noqa: E402

from methods_comparison._constants import CSRNET_FEATURE_DIM, CSRNET_FEATURE_LAYER  # noqa: E402


class CSRNetInfer:
    """Load a trained CSRNet checkpoint and run inference.

    Parameters
    ----------
    checkpoint_path:
        Path to the .pth file produced by train_csrnet.py
        (``methods_comparison/checkpoints/csrnet_best.pth``).
    device:
        Torch device string. ``None`` (default) auto-picks cuda > mps > cpu.
    """

    def __init__(self, checkpoint_path: Path, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # load_weights=True skips downloading VGG16 pretrained weights;
        # we restore everything from the checkpoint instead.
        self.model = CSRNet(load_weights=True).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(
            ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        )
        self.model.eval()

        self._tx = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        self._feat: np.ndarray | None = None

        hook_attached = False
        for name, mod in self.model.named_modules():
            if name == CSRNET_FEATURE_LAYER:
                mod.register_forward_hook(self._hook)
                hook_attached = True
                break
        if not hook_attached:
            raise RuntimeError(
                f"feature layer '{CSRNET_FEATURE_LAYER}' not found in CSRNet. "
                "Re-run preflight 0.4 to refresh _constants.py."
            )

    def _hook(self, _module, _inp, out):
        if out.dim() == 4:
            # Global-Average-Pool spatial dims -> shape (CSRNET_FEATURE_DIM,)
            self._feat = out.mean(dim=(2, 3)).detach().cpu().numpy()[0]
        else:
            self._feat = out.detach().cpu().numpy()[0]

    @torch.no_grad()
    def predict_count(self, image_path: Path) -> tuple[int, np.ndarray]:
        """Run inference on a single image.

        Parameters
        ----------
        image_path:
            Path to a PNG/JPG image (AoLP frame from Exp_09).

        Returns
        -------
        count:
            Estimated strip count (rounded sum of the predicted density map).
        feat:
            Feature vector of shape ``(CSRNET_FEATURE_DIM,)`` extracted via a
            forward hook on ``CSRNET_FEATURE_LAYER`` (Global-Average-Pool of
            the Conv2d output).
        """
        img = (
            self._tx(Image.open(image_path).convert("L"))
            .unsqueeze(0)
            .to(self.device)
        )
        density = self.model(img)
        count = int(density.sum().round().item())
        assert self._feat is not None, "forward hook did not fire"
        assert self._feat.shape == (CSRNET_FEATURE_DIM,), (
            f"unexpected feature shape {self._feat.shape}"
        )
        return count, self._feat.copy()
