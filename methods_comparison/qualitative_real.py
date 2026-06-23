"""Real-Frame-Eval; mode='a' = qualitative grid, mode='b' = quasi-quantitative MAE.

Usage:
    python -m methods_comparison.qualitative_real --mode a
    python -m methods_comparison.qualitative_real --mode b

Mode (a): 3 real frames, 3xK-Grid PNG, no quantitative MAE.
Mode (b): 20-30 real frames with GT annotations, MAE per method, scatter plots.
"""

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from methods_comparison._constants import MEASUREMENT_BOX
from methods_comparison.watershed import predict_count as watershed_predict, WatershedParams


# Real-frame paths for mode (a). Pre-resized to 512x512 by hand (PIL bilinear);
# see methods_comparison/real_frames/ for the source-rendering script. The hero
# frame is the AOP_15 figure from the prior Composites_AI_2025 publication.
FRAMES_MODE_A = [
    Path("methods_comparison/real_frames/lab_AOP_15.png"),
    Path("methods_comparison/real_frames/production_0meter.png"),
    Path("methods_comparison/real_frames/production_1k2meter.png"),
]


def _load_methods():
    """Load watershed parameters and optionally CSRNet."""
    wp = json.loads(Path("methods_comparison/watershed_params.json").read_text())
    wparams = WatershedParams(
        min_area=wp["min_area"],
        max_area=2000,
        min_distance=wp["min_distance"],
        measurement_box=MEASUREMENT_BOX,
        input_kind=wp.get("input_kind", "aolp")
    )

    csrnet = None
    csrnet_path = Path("methods_comparison/checkpoints/csrnet_best.pth")
    if csrnet_path.exists():
        try:
            from methods_comparison.csrnet_infer import CSRNetInfer
            csrnet = CSRNetInfer(csrnet_path)
        except Exception as e:
            print(f"Warning: CSRNet load failed ({e}). Mode (a) will use watershed + TTK only.")
    else:
        print("Warning: CSRNet checkpoint not found at methods_comparison/checkpoints/csrnet_best.pth")
        print("Mode (a) will use watershed + TTK only. CSRNet is a future enhancement.")

    return wparams, csrnet


def _ttk_count(img_path: Path) -> int:
    """Run TTK persistent-homology pipeline via pvpython.

    Converts BMP to temporary PNG if needed, since TTK's PNGSeriesReader
    may not natively support BMP.
    """
    cache_dir = Path("methods_comparison/cache/topology_real")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{img_path.stem}.json"

    if cache.exists():
        return int(json.loads(cache.read_text())["count"])

    # If input is BMP, convert to temp PNG for TTK
    input_path = img_path
    temp_png = None
    if img_path.suffix.lower() == ".bmp":
        temp_png = Path(tempfile.gettempdir()) / f"{img_path.stem}_ttk.png"
        img = Image.open(img_path).convert("L").resize((512, 512))
        img.save(temp_png)
        input_path = temp_png

    try:
        subprocess.run([
            "/Applications/ParaView-6.0.1.app/Contents/bin/pvpython",
            "methods_comparison/topology_ttk.py",
            "--input", str(input_path),
            "--persistence", "6.0",
            "--out", str(cache),
        ], check=True, capture_output=True)
    finally:
        if temp_png and temp_png.exists():
            temp_png.unlink()

    return int(json.loads(cache.read_text())["count"])


def main_qualitative(frames: list[Path], out: Path, wparams: WatershedParams, csrnet):
    """Generate 3xK qualitative grid PNG with watershed, CSRNet, TTK predictions."""
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, len(frames), figsize=(4 * len(frames), 10), squeeze=False)

    for col, frame_path in enumerate(frames):
        # Load image and rescale to 512x512 if needed
        img = Image.open(frame_path).convert("L").resize((512, 512))
        arr = np.asarray(img)

        # Watershed
        w_count = watershed_predict(frame_path, wparams)

        # CSRNet (if available)
        c_count = None
        if csrnet is not None:
            try:
                c_count, _ = csrnet.predict_count(frame_path)
            except Exception as e:
                print(f"Warning: CSRNet inference failed on {frame_path.name}: {e}")

        # TTK
        try:
            t_count = _ttk_count(frame_path)
        except Exception as e:
            print(f"Warning: TTK inference failed on {frame_path.name}: {e}")
            t_count = None

        # Row 0: Original image
        axs[0, col].imshow(arr, cmap="gray")
        axs[0, col].set_title(f"{frame_path.name}", fontsize=8)
        axs[0, col].axis("off")

        # Row 1: Watershed result
        axs[1, col].imshow(arr, cmap="gray")
        title_parts = [f"Watershed={w_count}"]
        if c_count is not None:
            title_parts.append(f"CSRNet={c_count}")
        if t_count is not None:
            title_parts.append(f"TTK={t_count}")
        axs[1, col].set_title(" ".join(title_parts), fontsize=9)
        axs[1, col].axis("off")

        # Row 2: Comparison text
        axs[2, col].axis("off")
        text_lines = [
            f"Watershed: {w_count}",
        ]
        if c_count is not None:
            text_lines.append(f"CSRNet: {c_count}")
        else:
            text_lines.append("CSRNet: (not available)")
        if t_count is not None:
            text_lines.append(f"TTK: {t_count}")
        else:
            text_lines.append("TTK: (failed)")

        axs[2, col].text(0.1, 0.5, "\n".join(text_lines), fontsize=10, verticalalignment="center")

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved qualitative grid to {out}")


def main_quantitative(annotations_path: Path, out_table: Path, out_fig: Path, wparams: WatershedParams, csrnet):
    """Generate MAE table and scatter plots for 20-30 annotated real frames."""
    if not annotations_path.exists():
        print(f"Error: annotation file {annotations_path} not found.")
        print("Mode (b) requires a JSON file with frame paths and ground-truth count estimates.")
        print("Format: {'/path/to/frame.bmp': <int>, ...}")
        print("See methods_comparison/real_annotations.json (example / template).")
        exit(1)

    annotations = json.loads(annotations_path.read_text())
    out_table.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame_str, gt in annotations.items():
        p = Path(frame_str)
        if not p.exists():
            print(f"Warning: frame {p} not found, skipping.")
            continue

        w_count = watershed_predict(p, wparams)
        c_count = None
        if csrnet is not None:
            try:
                c_count, _ = csrnet.predict_count(p)
            except Exception as e:
                print(f"Warning: CSRNet inference failed on {p.name}: {e}")

        try:
            t_count = _ttk_count(p)
        except Exception as e:
            print(f"Warning: TTK inference failed on {p.name}: {e}")
            t_count = None

        row = {
            "frame": p.name,
            "gt_count_est": int(gt),
            "watershed_pred": int(w_count),
        }
        if c_count is not None:
            row["csrnet_pred"] = int(c_count)
        if t_count is not None:
            row["ttk_pred"] = int(t_count)

        rows.append(row)

    if not rows:
        print("Error: no valid frames in annotation file.")
        exit(1)

    # Write CSV
    with out_table.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved MAE table to {out_table}")

    # Generate scatter plots
    import pandas as pd
    df = pd.DataFrame(rows)

    methods = [col for col in df.columns if col.endswith("_pred")]
    fig, axs = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5), sharex=True, sharey=True)
    if len(methods) == 1:
        axs = [axs]

    for ax, method_col in zip(axs, methods):
        ax.scatter(df["gt_count_est"], df[method_col], s=15)
        lo, hi = df["gt_count_est"].min(), df["gt_count_est"].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        mae = float((df[method_col] - df["gt_count_est"]).abs().mean())
        method_name = method_col.replace("_pred", "")
        ax.set_title(f"{method_name} MAE={mae:.1f}", fontsize=10)
        ax.set_xlabel("GT (estimated)")

    axs[0].set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300)
    print(f"Saved scatter plots to {out_fig}")


def read_mode_from_paper_claim() -> str:
    """Read the mode choice from paper_claim.md.

    Returns 'a' or 'b', defaulting to 'a' if not set or PENDING.
    """
    paper_claim = Path("methods_comparison/paper_claim.md")
    if not paper_claim.exists():
        print("Warning: paper_claim.md not found, defaulting to mode=a")
        return "a"

    text = paper_claim.read_text()
    for line in text.split("\n"):
        if "Choice:" in line:
            if "a" in line:
                return "a"
            elif "b" in line:
                return "b"

    print("Warning: A4 Real-Frame Strategy not set in paper_claim.md, defaulting to mode=a")
    return "a"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Real-frame evaluation: qualitative grid (mode=a) or quantitative MAE (mode=b)"
    )
    ap.add_argument("--mode", choices=("a", "b"), help="Evaluation mode")
    args = ap.parse_args()

    # Determine mode: CLI arg, or fall back to paper_claim.md, or default to 'a'
    mode = args.mode
    if mode is None:
        mode = read_mode_from_paper_claim()

    print(f"Running real-frame evaluation in mode={mode}")

    # Load methods
    wparams, csrnet = _load_methods()

    # Output directory
    fig_dir = Path(os.environ.get("FIGURE_OUTPUT", "./figures"))

    if mode == "a":
        main_qualitative(FRAMES_MODE_A, fig_dir / "fig3_real.png", wparams, csrnet)
    else:
        main_quantitative(
            Path("methods_comparison/real_annotations.json"),
            fig_dir / "table3_real_mae.csv",
            fig_dir / "fig3_real.pdf",
            wparams,
            csrnet
        )

    print("Done.")
