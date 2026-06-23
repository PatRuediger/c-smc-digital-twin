"""Qualitative Fig 4 from synthetic test-set predictions, with spatial overlays.

Picks one representative frame per density regime (median-GT row of each
regime in results.csv) and renders a 3-row x 3-column grid:

  row = density regime (low / medium / high)
  col 1: AoLP image with GT centroids (from YOLO-OBB bbox file) as green dots
  col 2: AoLP image with Watershed region outlines (red) and centroids
  col 3: AoLP image with CSRNet density-map heatmap overlay (jet, alpha=0.5)

Each row title states the regime, GT count, and all four method predictions
(Watershed, CSRNet, TTK, Fusion).

Output: 19_ECCM_Oslo_2026/draft/figures/fig4_qualitative.{png,pdf}

Run with:
    /path/to/ivw_dt_pytorch/bin/python3 -m methods_comparison.qualitative_synth
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage.segmentation import watershed

from methods_comparison.csrnet_infer import CSRNetInfer

FIG_DIR = Path(
    os.environ.get("OUTPUT_ROOT", "./output/")
    "19_ECCM_Oslo_2026/draft/figures"
)
RESULTS = Path("methods_comparison/results.csv")
SPLITS = Path("methods_comparison/splits.json")
WATERSHED_PARAMS = Path("methods_comparison/watershed_params.json")
CHECKPOINT = Path("methods_comparison/checkpoints/csrnet_best.pth")
TTK_QUAL_CACHE = Path("methods_comparison/cache/topology_qual")
REGIME_ORDER = ("low", "medium", "high")
IMAGE_RESOLUTION = 512


def _pick_median_per_regime() -> list[dict]:
    rows = list(csv.DictReader(RESULTS.open()))
    by_regime: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_regime[r["density_regime"]].append(r)
    chosen = []
    for regime in REGIME_ORDER:
        sub = by_regime.get(regime, [])
        if not sub:
            continue
        sub.sort(key=lambda r: int(r["gt_count"]))
        chosen.append(sub[len(sub) // 2])
    return chosen


def _resolve_image_path(image_id: str) -> Path:
    splits = json.loads(SPLITS.read_text())
    for entry in splits["test"]:
        if f"{entry['run_id']}_{entry['frame_id']}" == image_id:
            return Path(entry["image_path"])
    raise KeyError(f"image_id {image_id} not found in test split")


def _bbox_path_for(image_path: Path) -> Path:
    """Derive bbox.txt for an AoLP image (same convention as prepare_csrnet_dataset)."""
    stem = image_path.stem.replace("_aolp", "")
    run_dir = image_path.parents[3]
    return run_dir / "imgs" / f"{stem}.txt"


def _read_gt_centroids(image_path: Path) -> np.ndarray:
    """YOLO-OBB lines -> N x 2 array of (x, y) pixel centroids."""
    bbox = _bbox_path_for(image_path)
    coords = []
    with bbox.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cx_norm, cy_norm = float(parts[1]), float(parts[2])
            coords.append((cx_norm * IMAGE_RESOLUTION, cy_norm * IMAGE_RESOLUTION))
    return np.array(coords) if coords else np.empty((0, 2))


def _watershed_overlay(arr: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the watershed pipeline and return (labels, centroids)."""
    thr = threshold_otsu(arr)
    binary = arr > thr
    distance = np.asarray(ndi.distance_transform_edt(binary))
    coords = peak_local_max(distance, min_distance=params["min_distance"], labels=binary)
    markers = np.zeros(distance.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    labels = watershed(-distance, markers, mask=binary)
    centroids = []
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        mask = labels == lbl
        area = int(mask.sum())
        if not (params["min_area"] <= area <= 2000):
            continue
        ys, xs = np.where(mask)
        centroids.append((xs.mean(), ys.mean()))
    return labels, np.array(centroids) if centroids else np.empty((0, 2))


def _csrnet_density(image_path: Path, csrnet: CSRNetInfer) -> np.ndarray:
    """Run CSRNet and return the upsampled-to-IMAGE_RESOLUTION density map."""
    img = csrnet._tx(Image.open(image_path).convert("L")).unsqueeze(0).to(csrnet.device)
    with torch.no_grad():
        density = csrnet.model(img)
    dmap = density.squeeze().cpu().numpy()
    img_pil = Image.fromarray(dmap).resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION), Image.Resampling.BILINEAR)
    return np.asarray(img_pil)


def main() -> None:
    chosen = _pick_median_per_regime()
    if len(chosen) != 3:
        raise RuntimeError(f"expected 3 regimes, got {len(chosen)}")

    wp = json.loads(WATERSHED_PARAMS.read_text())
    csrnet = CSRNetInfer(CHECKPOINT)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    col_titles = ["Ground truth", "Watershed", "CSRNet density", "Morse-Smale complex"]

    for row_idx, row in enumerate(chosen):
        img_path = _resolve_image_path(row["image_id"])
        arr = np.asarray(Image.open(img_path).convert("L"))

        gt_centroids = _read_gt_centroids(img_path)
        ws_labels, ws_centroids = _watershed_overlay(arr, wp)
        cs_density = _csrnet_density(img_path, csrnet)

        # TTK centroids loaded from pre-computed cache
        ttk_cache = TTK_QUAL_CACHE / f"{row['image_id']}.json"
        ttk_centroids = np.empty((0, 2))
        ttk_raw = 0
        if ttk_cache.exists():
            payload = json.loads(ttk_cache.read_text())
            ttk_raw = int(payload.get("count", 0))
            ttk_centroids = np.array(payload.get("centroids", []) or []).reshape(-1, 2)

        gt = int(row["gt_count"])
        w_count = int(row["watershed_pred"])
        c_count = int(row["csrnet_pred"])
        t_count = int(row["ttk_pred"])
        f_count = int(row["fusion_pred"])
        regime = row["density_regime"]

        # Col 0: GT centroids on AoLP
        ax = axes[row_idx, 0]
        ax.imshow(arr, cmap="gray")
        if len(gt_centroids):
            ax.scatter(gt_centroids[:, 0], gt_centroids[:, 1],
                       s=8, c="lime", marker="o", linewidths=0.4, edgecolors="black", alpha=0.8)
        ax.set_xlim(0, IMAGE_RESOLUTION)
        ax.set_ylim(IMAGE_RESOLUTION, 0)
        ax.axis("off")

        # Col 1: Watershed region outlines on AoLP
        ax = axes[row_idx, 1]
        ax.imshow(arr, cmap="gray")
        for lbl in np.unique(ws_labels):
            if lbl == 0:
                continue
            mask = ws_labels == lbl
            for contour in find_contours(mask.astype(float), 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color="red", linewidth=0.5, alpha=0.7)
        if len(ws_centroids):
            ax.scatter(ws_centroids[:, 0], ws_centroids[:, 1],
                       s=6, c="red", marker="x", linewidths=0.6)
        ax.set_xlim(0, IMAGE_RESOLUTION)
        ax.set_ylim(IMAGE_RESOLUTION, 0)
        ax.axis("off")

        # Col 2: CSRNet density heatmap overlay
        ax = axes[row_idx, 2]
        ax.imshow(arr, cmap="gray")
        ax.imshow(cs_density, cmap="jet", alpha=0.5,
                  extent=(0, IMAGE_RESOLUTION, IMAGE_RESOLUTION, 0))
        ax.axis("off")

        # Col 3: TTK Morse-Smale centroid density (2D histogram of over-segmented
        # ascending-manifold centres; the heatmap parallels the CSRNet density
        # column so that the topological "where does TTK think strips are"
        # signal is directly comparable to the learning-based one).
        ax = axes[row_idx, 3]
        ax.imshow(arr, cmap="gray")
        if len(ttk_centroids):
            n_bins = 32
            hist, _, _ = np.histogram2d(
                ttk_centroids[:, 1], ttk_centroids[:, 0],
                bins=n_bins,
                range=[[0, IMAGE_RESOLUTION], [0, IMAGE_RESOLUTION]],
            )
            ax.imshow(
                hist, cmap="Oranges", alpha=0.55,
                extent=(0, IMAGE_RESOLUTION, IMAGE_RESOLUTION, 0),
                interpolation="bilinear",
            )
        ax.set_xlim(0, IMAGE_RESOLUTION)
        ax.set_ylim(IMAGE_RESOLUTION, 0)
        ax.axis("off")

        # Row title (left of col 0)
        msc_label = f"M={t_count}" if ttk_raw == 0 else f"M={t_count} (raw {ttk_raw})"
        axes[row_idx, 0].set_ylabel(
            f"{regime}\nGT={gt}\nW={w_count}  C={c_count}\n{msc_label}\nF={f_count}",
            rotation=0, ha="right", va="center", fontsize=13, labelpad=50,
        )
        axes[row_idx, 0].axis("on")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        for spine in axes[row_idx, 0].spines.values():
            spine.set_visible(False)

    # Column titles
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=14, pad=10)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "fig4_qualitative.png"
    out_pdf = FIG_DIR / "fig4_qualitative.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS: wrote {out_png}  ({out_png.stat().st_size:,} bytes)")
    print(f"      wrote {out_pdf}  ({out_pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
