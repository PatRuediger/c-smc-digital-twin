import csv
import json
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not Path("methods_comparison/checkpoints/csrnet_best.pth").exists()
    or not Path("methods_comparison/cache/topology").exists(),
    reason="CSRNet checkpoint or TTK test cache missing",
)
def test_aggregate_writes_csv_with_all_method_columns(tmp_path: Path):
    from methods_comparison.run_eval import aggregate_results

    out = tmp_path / "results.csv"
    aggregate_results(
        splits_file=Path("methods_comparison/splits.json"),
        split_name="test",
        watershed_params=Path("methods_comparison/watershed_params.json"),
        csrnet_ckpt=Path("methods_comparison/checkpoints/csrnet_best.pth"),
        topology_cache=Path("methods_comparison/cache/topology"),
        out=out,
    )
    rows = list(csv.DictReader(out.open()))
    splits = json.loads(Path("methods_comparison/splits.json").read_text())
    assert len(rows) == len(splits["test"])
    for col in ("image_id", "gt_count", "density_regime",
                "watershed_pred", "csrnet_pred", "ttk_pred"):
        assert col in rows[0]
