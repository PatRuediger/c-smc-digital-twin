"""Prepare real-frame PNGs for qualitative_real.py by cropping to the material ROI.

Production frames from the line camera capture the full belt width (744x1028);
the material area occupies only rows 670-1028. Without cropping, Watershed and
CSRNet count belt texture and machine structure, not strips.

The lab frame (CleanAOP_15.tif) is already material-dominated and gets a
center-square crop to remove the dark corners from the lab setup.

Usage:
    python -m methods_comparison.prepare_real_frames
"""

from pathlib import Path

import numpy as np
from PIL import Image

from methods_comparison._constants import PRODUCTION_MATERIAL_ROI, RAW_FRAME_SOURCES

DATA_ROOT = Path(__file__).resolve().parent.parent.parent  # comp-data-synth/
OUT_DIR = Path(__file__).resolve().parent / "real_frames"
TARGET_SIZE = 512


def _crop_production(raw: np.ndarray) -> np.ndarray:
    y0, x0, y1, x1 = PRODUCTION_MATERIAL_ROI
    return raw[y0:y1, x0:x1]


def _crop_lab(raw: np.ndarray) -> np.ndarray:
    """Center-square crop: take the largest centered square from the lab frame."""
    h, w = raw.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return raw[y0:y0 + side, x0:x0 + side]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, rel_path in RAW_FRAME_SOURCES.items():
        src = DATA_ROOT / rel_path
        if not src.exists():
            print(f"SKIP {name}: source not found at {src}")
            continue

        raw = np.array(Image.open(src).convert("L"))
        print(f"{name}: raw {raw.shape}")

        if name.startswith("production_"):
            cropped = _crop_production(raw)
        else:
            cropped = _crop_lab(raw)

        print(f"  cropped -> {cropped.shape}")
        out = Image.fromarray(cropped).resize(
            (TARGET_SIZE, TARGET_SIZE), Image.BILINEAR
        )
        out_path = OUT_DIR / f"{name}.png"
        out.save(out_path)
        print(f"  saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
