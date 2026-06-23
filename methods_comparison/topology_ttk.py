"""TTK persistent-homology pipeline for strip counting via pvpython.

Run exclusively via pvpython (NOT python3):
    /Applications/ParaView-6.0.1.app/Contents/bin/pvpython \\
        methods_comparison/topology_ttk.py \\
        --input <path/to/aolp.png> \\
        --persistence <threshold_float> \\
        --out <path/to/output.json>

Output JSON schema:
    {
        "count": <int>,                  # ascending-manifold region count
        "persistence_summary": {
            "n_pairs": <int>,
            "max_persistence": <float>,
            "total_persistence": <float>,
            "persistence_entropy": <float>,
            "betti_dim0": <int>,
            "betti_dim1": <int>
        },
        "persistence_threshold": <float>,
        "input": <str>
    }

Pipeline:
    PNGSeriesReader
        -> Calculator (extract scalar channel from PNGImage)
        -> TTKPersistenceDiagram  (compute summary on unsmoothed field)
        -> TTKTopologicalSimplificationByPersistence (remove low-persistence features)
        -> TTKMorseSmaleComplex (ascending-manifold segmentation -> strip count)
"""
import argparse
import json
from pathlib import Path

TTK_PLUGIN = (
    "/Applications/ParaView-6.0.1.app/Contents/Plugins/TopologyToolKit/TopologyToolKit.so"
)

# ---------------------------------------------------------------------------
# Imports from paraview.simple (must be done BEFORE connecting / loading plugin)
# ---------------------------------------------------------------------------
from paraview.simple import (  # noqa: E402
    Connect,
    LoadPlugin,
    PNGSeriesReader,
    Calculator,
)
from paraview import servermanager  # noqa: E402

import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_persistence(diag_proxy) -> dict:
    """Extract summary statistics from a ttkPersistenceDiagram proxy.

    In ParaView 6.0.1 / TTK 2.0.x the persistence diagram output is an
    unstructured grid whose CELL data carries the relevant arrays:
        - "Persistence"   : float, one value per pair
        - "PairType"      : int, 0 = min-saddle, 1 = sad-sad, 2 = sad-max
    """
    d = servermanager.Fetch(diag_proxy)
    cell_data = d.GetCellData()
    pers_arr = cell_data.GetArray("Persistence")
    type_arr = cell_data.GetArray("PairType")

    if pers_arr is None or pers_arr.GetNumberOfTuples() == 0:
        return {
            "n_pairs": 0,
            "max_persistence": 0.0,
            "total_persistence": 0.0,
            "persistence_entropy": 0.0,
            "betti_dim0": 0,
            "betti_dim1": 0,
        }

    n = pers_arr.GetNumberOfTuples()
    pers = np.array([pers_arr.GetValue(i) for i in range(n)], dtype=float)
    dims = np.array([type_arr.GetValue(i) for i in range(n)], dtype=int) if type_arr is not None else np.zeros(n, dtype=int)

    # Keep only finite positive persistence values (exclude the infinite pair)
    mask = (pers > 0) & np.isfinite(pers)
    pers = pers[mask]
    dims = dims[mask]

    if pers.size == 0:
        return {
            "n_pairs": 0,
            "max_persistence": 0.0,
            "total_persistence": 0.0,
            "persistence_entropy": 0.0,
            "betti_dim0": 0,
            "betti_dim1": 0,
        }

    total = float(pers.sum())
    p = pers / total if total > 0 else pers
    entropy = float(-(p * np.log(p + 1e-12)).sum())

    return {
        "n_pairs": int(pers.size),
        "max_persistence": float(pers.max()),
        "total_persistence": total,
        "persistence_entropy": entropy,
        "betti_dim0": int((dims == 0).sum()),
        "betti_dim1": int((dims == 1).sum()),
    }


def _ascending_manifolds(msc_proxy) -> tuple[int, list[list[float]]]:
    """Return (count, centroids_xy) for ascending-manifold regions.

    The segmentation is on output port 3 (idx=3). Each pixel carries an
    'AscendingManifold' label; -1 marks pixels not assigned to any region
    (separatrices / boundary). Centroids are mean (x, y) per region in image
    pixel coordinates.
    """
    seg = servermanager.Fetch(msc_proxy, idx=3)
    arr = seg.GetPointData().GetArray("AscendingManifold")
    if arr is None:
        return 0, []
    n_pts = arr.GetNumberOfTuples()
    vals = np.array([arr.GetValue(i) for i in range(n_pts)], dtype=int)

    # Reconstruct (x, y) coordinates from the structured grid extents
    dims = seg.GetDimensions()
    nx, ny = dims[0], dims[1]
    xs = np.arange(nx)
    ys = np.arange(ny)
    xx, yy = np.meshgrid(xs, ys)
    xs_flat = xx.flatten()
    ys_flat = yy.flatten()
    if len(xs_flat) != n_pts:
        return 0, []

    centroids = []
    unique = np.unique(vals[vals >= 0])
    for lbl in unique:
        mask = vals == lbl
        cx = float(xs_flat[mask].mean())
        cy = float(ys_flat[mask].mean())
        centroids.append([cx, cy])
    return int(unique.size), centroids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="TTK persistence-diagram pipeline for strip counting"
    )
    ap.add_argument("--input", required=True, help="Path to AoLP PNG image")
    ap.add_argument("--persistence", type=float, required=True, help="Persistence threshold (absolute)")
    ap.add_argument("--out", required=True, help="Output JSON file path")
    args = ap.parse_args()

    # Connect to in-process ParaView server and load TTK plugin
    Connect()
    LoadPlugin(TTK_PLUGIN, remote=False)

    # Deferred import of TTK filters (only available after LoadPlugin)
    import paraview.simple as pvsimple
    TTKPersistenceDiagram = pvsimple.TTKPersistenceDiagram
    TTKTopologicalSimplificationByPersistence = pvsimple.TTKTopologicalSimplificationByPersistence
    TTKMorseSmaleComplex = pvsimple.TTKMorseSmaleComplex

    # 1. Read image
    img = PNGSeriesReader(FileNames=[args.input])

    # 2. Extract scalar channel (AoLP images are 16-bit grayscale stored as
    #    3-component PNGImage; all channels are identical -- take component 0)
    scalar_field = Calculator(Input=img)
    scalar_field.Function = "PNGImage_X"
    scalar_field.ResultArrayName = "scalar"
    scalar_field.AttributeType = "Point Data"

    # 3. Compute persistence diagram on the raw scalar field (for summary stats)
    diag = TTKPersistenceDiagram(Input=scalar_field)
    diag.ScalarField = ["POINTS", "scalar"]
    diag.UpdatePipeline()
    summary = _summarize_persistence(diag)

    # 4. Topological simplification: remove features below persistence threshold
    simp = TTKTopologicalSimplificationByPersistence(Input=scalar_field)
    simp.PersistenceThreshold = args.persistence
    simp.ThresholdIsAbsolute = 1
    simp.InputArray = ["POINTS", "scalar"]

    # 5. Morse-Smale complex: ascending-manifold segmentation = strip regions
    msc = TTKMorseSmaleComplex(Input=simp)
    msc.ScalarField = "scalar"
    msc.AscendingSegmentation = 1
    msc.UpdatePipeline()

    count, centroids = _ascending_manifolds(msc)

    # 6. Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "count": count,
                "centroids": centroids,
                "persistence_summary": summary,
                "persistence_threshold": args.persistence,
                "input": args.input,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
