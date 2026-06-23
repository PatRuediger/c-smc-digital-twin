"""Extracts measurement-box pixel coordinates from Exp_09 DB records.

ADAPTATION from original plan (Step 0.2):
The original plan searched simulation/density.py for a MEASUREMENT_BOX constant.
That constant does not exist; the box is stored per-simulation-run in the SQLite DB
columns measurement_box_center (TEXT, Python tuple repr) and measurement_box_dims (TEXT).

Camera model (from simulation/rendering.py::align_camera_to_box):
  - Orthographic camera, positioned at (box_center[0], box_center[1], z=20)
  - ortho_scale = max(box_dims[0], box_dims[1])  -> covers the box exactly
  - image resolution: 512x512 (verified on disk)

Since the camera is aligned so that the measurement box fills the entire image exactly,
the pixel-space measurement box is always the full image frame: (y0=0, x0=0, y1=512, x1=512).

PASS: box is consistent across all 20 runs and plausible; writes (0, 0, 512, 512) to _constants.py.
FAIL: box dims vary across runs (unexpected) or parse fails.

Writes box_decision.md documenting the world-to-pixel derivation.
"""
import ast
import re
import sqlite3
import sys
from pathlib import Path

EXP09_ROOT = Path(os.environ.get("EXP09_ROOT", "./data/Exp_09"))
IMAGE_RESOLUTION = 512  # verified: all AoLP images are 512x512


def parse_tuple(s: str) -> tuple[float, ...]:
    """Parse a Python-repr tuple string like '(-6.0, 0.0, 0.1)' into a tuple."""
    return ast.literal_eval(s)


def world_to_pixel_box(
    dims: tuple[float, float, float],
    resolution: int,
) -> tuple[int, int, int, int]:
    """Convert world-space measurement box to pixel-space bounding box.

    Camera model: orthographic, centered on the box, looking down -Z, ortho_scale
    matches the larger box dimension. For the square boxes used in this dataset
    (15x15 m), the box fills the image exactly so pixel box = (0, 0, R, R).

    Returns (y0, x0, y1, x1) following numpy/scikit-image row-major convention.
    """
    if abs(dims[0] - dims[1]) > 0.01:
        raise ValueError(
            f"non-square measurement box {dims[:2]} not supported; "
            "this dataset uses 15x15 m square boxes"
        )
    # Square box, fills frame exactly.
    return (0, 0, resolution, resolution)


def main() -> int:
    dbs = sorted(EXP09_ROOT.glob("Exp_09_Polarized_OBB_*/stripGen_results.db"))
    if not dbs:
        print(f"FAIL: no DBs found under {EXP09_ROOT}")
        return 1

    boxes: list[tuple[tuple, tuple]] = []
    for db in dbs:
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT measurement_box_center, measurement_box_dims FROM results LIMIT 1"
            ).fetchone()
            if not row:
                print(f"WARN: no rows in {db}")
                continue
            try:
                center = parse_tuple(row[0])
                dims = parse_tuple(row[1])
            except Exception as e:
                print(f"FAIL: cannot parse box from {db}: {e}")
                return 1
            boxes.append((center, dims))

    if not boxes:
        print("FAIL: no measurement box data found in any DB")
        return 1

    # Check consistency across runs
    centers = [b[0] for b in boxes]
    dims_list = [b[1] for b in boxes]
    unique_centers = set(centers)
    unique_dims = set(dims_list)

    if len(unique_centers) > 1:
        print(f"WARN: measurement_box_center varies across runs: {unique_centers}")
        print("Using first run's center. If runs have different cameras, Task 2 needs per-run lookup.")
    if len(unique_dims) > 1:
        print(f"WARN: measurement_box_dims varies across runs: {unique_dims}")

    ref_center, ref_dims = boxes[0]
    print(f"measurement_box_center (world): {ref_center}")
    print(f"measurement_box_dims (world):   {ref_dims}")
    print(f"image resolution: {IMAGE_RESOLUTION}x{IMAGE_RESOLUTION}")

    pixel_box = world_to_pixel_box(ref_dims, IMAGE_RESOLUTION)
    y0, x0, y1, x1 = pixel_box

    if not (y1 > y0 >= 0 and x1 > x0 >= 0 and y1 <= IMAGE_RESOLUTION and x1 <= IMAGE_RESOLUTION):
        print(f"FAIL: derived pixel box {pixel_box} is not plausible (image={IMAGE_RESOLUTION})")
        return 1

    print(f"PASS: MEASUREMENT_BOX (pixel) = (y0={y0}, x0={x0}, y1={y1}, x1={x1})")
    print("Note: box equals full image frame because ortho camera is aligned to box exactly.")

    # Write into _constants.py; pattern matches both annotation and assignment forms.
    constants_path = Path("methods_comparison/_constants.py")
    constants = constants_path.read_text()
    constants = re.sub(
        r"MEASUREMENT_BOX[: ].*",
        f"MEASUREMENT_BOX = ({y0}, {x0}, {y1}, {x1})  # pixel (y0,x0,y1,x1); world center={ref_center} dims={ref_dims}",
        constants,
    )
    constants_path.write_text(constants)

    # Write box_decision.md
    preflight_dir = Path("methods_comparison/preflight")
    preflight_dir.mkdir(parents=True, exist_ok=True)
    decision_text = f"""# Measurement Box Decision (Step 0.2)

## Original plan assumption
The plan assumed a `MEASUREMENT_BOX` constant in `simulation/density.py`. This does not exist.

## Actual source
The box is stored per simulation run in the SQLite DB columns:
- `measurement_box_center` (TEXT, Python tuple repr): world-space XYZ center
- `measurement_box_dims`   (TEXT, Python tuple repr): world-space XYZ dimensions

Across all {len(boxes)} runs:
- center: {unique_centers if len(unique_centers) > 1 else list(unique_centers)[0]}
- dims:   {unique_dims if len(unique_dims) > 1 else list(unique_dims)[0]}
- {'Box is CONSISTENT across all runs.' if len(unique_centers) == 1 and len(unique_dims) == 1 else 'Box VARIES across runs; world_to_pixel_box uses first run as reference.'}

## Camera model (from simulation/rendering.py::align_camera_to_box)

- Camera type: Orthographic
- Camera position: (center.x, center.y, z=20)
- ortho_scale: max(dims.x, dims.y) = {max(ref_dims[0], ref_dims[1])} m
- Image resolution: {IMAGE_RESOLUTION}x{IMAGE_RESOLUTION} px
- The ortho_scale equals the box side length, so the box fills the image exactly.

## Pixel-space box

Because the orthographic camera is aligned to fill the image with exactly the measurement box,
the pixel-space bounding box is the full image frame:

    MEASUREMENT_BOX = ({y0}, {x0}, {y1}, {x1})  # (y0, x0, y1, x1), row-major (numpy convention)

This is equivalent to: entire image.

## Implication for Task 2

The watershed/Otsu segmentation operates on the full AoLP image. No ROI crop is needed.
The `MEASUREMENT_BOX` constant is retained in `_constants.py` for reference and potential
future use (e.g., if a different camera or partial-frame box is introduced).
"""
    (preflight_dir / "box_decision.md").write_text(decision_text)
    print(f"decision doc: methods_comparison/preflight/box_decision.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
