import csv
from pathlib import Path

import pytest

from methods_comparison.figures import main


@pytest.fixture
def fake_results(tmp_path: Path) -> Path:
    p = tmp_path / "results.csv"
    rows = []
    for i in range(30):
        gt = (i * 17) % 600 + 50
        regime = "low" if gt <= 220 else ("medium" if gt <= 375 else "high")
        rows.append({
            "image_id": f"run_{i}_frame_{i*10}",
            "run_id": f"run_{i % 5}",
            "frame_id": str(i * 10),
            "gt_count": str(gt),
            "density_regime": regime,
            "watershed_pred": str(gt + ((i * 7) % 41) - 20),
            "csrnet_pred": str(gt + ((i * 5) % 21) - 10),
            "ttk_pred": str(gt + ((i * 3) % 31) - 15),
            "fusion_pred": str(gt + ((i * 2) % 11) - 5),
        })
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def test_figures_produce_all_artifacts(fake_results, tmp_path: Path):
    out_dir = tmp_path / "figures_out"
    main(fake_results, out_dir)
    assert (out_dir / "table1_metrics.csv").exists()
    assert (out_dir / "table2_paired_diffs.csv").exists()
    assert (out_dir / "fig2_quantitative.pdf").exists()
    assert (out_dir / "fig3_scatter.pdf").exists()
    # quick non-empty check
    assert (out_dir / "fig2_quantitative.pdf").stat().st_size > 1000
