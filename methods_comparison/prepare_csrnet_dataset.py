"""Build the dataset layout that csrnet/csrnet_dataset.py expects.

Reads methods_comparison/splits.json (Task 1 output) and produces:

  methods_comparison/csrnet_data/
    images/
      train/<run_id>__<filename>.png   (symlinks to Exp_09 AoLP PNGs)
      val/<run_id>__<filename>.png
    points_annotations/
      train/<run_id>__<stem>.csv       (columns: x, y in pixel coords)
      val/<run_id>__<stem>.csv

Strip centroids come from the YOLO-OBB bbox.txt file co-located with each frame,
denormalized from [0,1] to pixel coords. Image resolution is read from _constants.py.

Idempotent: re-running overwrites symlinks + CSVs. Safe to invoke multiple times.

Run with:
    python3 -m methods_comparison.prepare_csrnet_dataset
"""
import csv
import json
from pathlib import Path

EXP09_ROOT = Path(os.environ.get("EXP09_ROOT", "./data/Exp_09"))
SPLITS = Path("methods_comparison/splits.json")
OUT_ROOT = Path("methods_comparison/csrnet_data")
IMAGE_RESOLUTION = 512


def _bbox_path_for(image_path: Path) -> Path:
    """Derive the bbox.txt path from an AoLP PNG path.

    AoLP path:  .../<run>/imgs/polarization_output/aolp/<TS>_<HASH>_seed<N>_frame<NNNN>_aolp.png
    Bbox path:  .../<run>/imgs/<TS>_<HASH>_seed<N>_frame<NNNN>.txt
    """
    stem = image_path.stem.replace("_aolp", "")
    run_dir = image_path.parents[3]
    return run_dir / "imgs" / f"{stem}.txt"


def _read_centroids(bbox_file: Path) -> list[tuple[float, float]]:
    """Read YOLO-OBB lines, return list of (x_pixel, y_pixel) centroids."""
    centroids = []
    with bbox_file.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cx_norm = float(parts[1])
            cy_norm = float(parts[2])
            centroids.append((cx_norm * IMAGE_RESOLUTION, cy_norm * IMAGE_RESOLUTION))
    return centroids


def _materialize_split(split_entries: list[dict], split_name: str) -> int:
    images_dir = OUT_ROOT / "images" / split_name
    points_dir = OUT_ROOT / "points_annotations" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for entry in split_entries:
        image_path = Path(entry["image_path"])
        run_id = entry["run_id"]

        unique_stem = f"{run_id}__{image_path.stem}"
        link_target = images_dir / f"{unique_stem}.png"
        if link_target.exists() or link_target.is_symlink():
            link_target.unlink()
        link_target.symlink_to(image_path)

        bbox_file = _bbox_path_for(image_path)
        if not bbox_file.exists():
            print(f"WARN: missing bbox file {bbox_file}; skipping centroids for {unique_stem}")
            centroids = []
        else:
            centroids = _read_centroids(bbox_file)

        csv_target = points_dir / f"{unique_stem}.csv"
        with csv_target.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for x, y in centroids:
                writer.writerow([f"{x:.4f}", f"{y:.4f}"])
        n += 1

    print(f"  {split_name}: {n} frames materialized")
    return n


def main() -> int:
    if not SPLITS.exists():
        print(f"FAIL: {SPLITS} missing. Run Task 1 (build_splits) first.")
        return 1

    splits = json.loads(SPLITS.read_text())
    print(f"Materializing CSRNet dataset layout under {OUT_ROOT}")
    n_train = _materialize_split(splits["train"], "train")
    n_val = _materialize_split(splits["val"], "val")
    print(f"PASS: train={n_train}, val={n_val}; ready for csrnet/train_csrnet.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
