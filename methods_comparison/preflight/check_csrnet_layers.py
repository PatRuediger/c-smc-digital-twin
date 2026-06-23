"""Druckt alle Layer von CSRNet und schlaegt einen 512-Channel-Feature-Layer vor.

PASS: ein Layer mit Output-Channels == 512 existiert; Name wird in _constants.py
      als CSRNET_FEATURE_LAYER gesetzt.
FAIL: kein 512-Channel-Layer; alternative Dim wird vorgeschlagen.

Requires torch. If torch is not in the current interpreter, the script re-execs
using the known project venv at ivw_dt_pytorch.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

PYTORCH_VENV_PYTHON = Path(
    "python3"
)


def _ensure_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _reexec_with_venv() -> None:
    """Re-execute this script using the pytorch venv if torch is missing."""
    if not PYTORCH_VENV_PYTHON.exists():
        print(f"FAIL: torch not found and venv not at {PYTORCH_VENV_PYTHON}")
        sys.exit(1)
    # Pass the cwd via env so the re-invoked process uses the right working dir.
    env = dict(os.environ)
    env["PREFLIGHT_REEXEC"] = "1"
    result = subprocess.run(
        [str(PYTORCH_VENV_PYTHON), "-m", "methods_comparison.preflight.check_csrnet_layers"],
        env=env,
    )
    sys.exit(result.returncode)


def main() -> int:
    if not _ensure_torch():
        if os.environ.get("PREFLIGHT_REEXEC") == "1":
            print("FAIL: torch still not importable even after re-exec with venv python.")
            return 1
        print(f"torch not in current interpreter; re-execing with {PYTORCH_VENV_PYTHON}")
        _reexec_with_venv()  # exits

    import torch
    from torch import nn

    # adjust import to actual layout
    sys.path.insert(0, str(Path("csrnet").resolve()))
    try:
        from csrnet_model import CSRNet
    except ImportError as e:
        print(f"FAIL: cannot import CSRNet: {e}")
        return 1

    model = CSRNet(load_weights=False)
    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            candidates.append((name, mod.out_channels))
    if not candidates:
        print("FAIL: no Conv2d layers found")
        return 1

    print("Conv2d layers:")
    for name, ch in candidates:
        print(f"  {name}: out_channels={ch}")

    # prefer last 512-channel conv as feature (aligns with CSRNet paper: frontend output is 512-ch)
    f512 = [(n, c) for n, c in candidates if c == 512]
    if f512:
        chosen_name = f512[-1][0]
        chosen_dim = 512
        print(f"PASS: selected last 512-channel layer: '{chosen_name}'")
    else:
        # pick the deepest backbone-ish layer
        chosen_name, chosen_dim = candidates[len(candidates) // 2]
        print(f"WARN: no 512-channel layer; falling back to '{chosen_name}' ({chosen_dim} channels)")
        print("Task 6 Late-Fusion head should use Linear(CSRNET_FEATURE_DIM + 5, 64).")

    constants_path = Path("methods_comparison/_constants.py")
    constants = constants_path.read_text()
    constants = re.sub(
        r"CSRNET_FEATURE_LAYER[: ].*",
        f'CSRNET_FEATURE_LAYER = "{chosen_name}"',
        constants,
    )
    constants = re.sub(
        r"CSRNET_FEATURE_DIM[: ].*",
        f"CSRNET_FEATURE_DIM = {chosen_dim}",
        constants,
    )
    constants_path.write_text(constants)
    print(f"PASS: feature layer = '{chosen_name}', dim = {chosen_dim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
