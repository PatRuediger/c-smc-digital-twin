"""Aggregate watershed/CSRNet/TTK predictions into results.csv."""

import csv
import json
from pathlib import Path

from methods_comparison.data import load_split
from methods_comparison.watershed import predict_count as watershed_predict, WatershedParams
from methods_comparison.csrnet_infer import CSRNetInfer


def _load_ttk_calibration() -> tuple[float, float] | None:
    """Load TTK linear calibration coefficients if they exist.

    Returns (a, b) where ttk_pred_calibrated = a * raw + b.
    If calibration file missing, return None and let the caller decide behavior.
    """
    p = Path("methods_comparison/topology_calibration.json")
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return float(c["a"]), float(c["b"])


def aggregate_results(
    splits_file: Path,
    split_name: str,
    watershed_params: Path,
    csrnet_ckpt: Path,
    topology_cache: Path,
    out: Path,
) -> None:
    """Write a CSV with predictions from all three methods.

    Parameters
    ----------
    splits_file
        Path to methods_comparison/splits.json
    split_name
        "train", "val", or "test"
    watershed_params
        Path to methods_comparison/watershed_params.json
    csrnet_ckpt
        Path to methods_comparison/checkpoints/csrnet_best.pth
    topology_cache
        Path to methods_comparison/cache/topology/
    out
        Output CSV file path
    """
    # Load watershed params
    wp = json.loads(watershed_params.read_text())
    # Get measurement box from constants
    from methods_comparison._constants import MEASUREMENT_BOX
    box = MEASUREMENT_BOX
    wparams = WatershedParams(
        min_area=wp["min_area"],
        max_area=wp.get("max_area", 2000),
        min_distance=wp["min_distance"],
        measurement_box=box,
    )

    # Load CSRNet model
    csrnet = CSRNetInfer(csrnet_ckpt)

    # Load TTK calibration if available
    ttk_calib = _load_ttk_calibration()
    if ttk_calib is None:
        print("WARN: no topology_calibration.json; ttk_pred uses raw MSC count (likely ~18x GT)")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "image_id", "run_id", "frame_id",
                "gt_count", "density_regime",
                "watershed_pred", "csrnet_pred", "ttk_pred",
            ],
        )
        w.writeheader()

        for e in load_split(splits_file, split_name):
            # Watershed prediction
            ws_pred = watershed_predict(Path(e.image_path), wparams)

            # CSRNet prediction
            cs_pred, _ = csrnet.predict_count(Path(e.image_path))

            # TTK prediction from JSON cache
            ttk_path = topology_cache / f"{e.run_id}_{e.frame_id}.json"
            ttk_payload = json.loads(ttk_path.read_text())
            raw_ttk = int(ttk_payload["count"])

            # Apply calibration if available
            if ttk_calib is not None:
                a, b = ttk_calib
                ttk_pred = max(0, int(round(a * raw_ttk + b)))
            else:
                ttk_pred = raw_ttk

            w.writerow({
                "image_id": f"{e.run_id}_{e.frame_id}",
                "run_id": e.run_id,
                "frame_id": e.frame_id,
                "gt_count": e.gt_count,
                "density_regime": e.density_regime,
                "watershed_pred": ws_pred,
                "csrnet_pred": cs_pred,
                "ttk_pred": ttk_pred,
            })


if __name__ == "__main__":
    aggregate_results(
        splits_file=Path("methods_comparison/splits.json"),
        split_name="test",
        watershed_params=Path("methods_comparison/watershed_params.json"),
        csrnet_ckpt=Path("methods_comparison/checkpoints/csrnet_best.pth"),
        topology_cache=Path("methods_comparison/cache/topology"),
        out=Path("methods_comparison/results.csv"),
    )
