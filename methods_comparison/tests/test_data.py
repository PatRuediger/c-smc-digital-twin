import json
from collections import Counter
from pathlib import Path

from methods_comparison.data import build_splits, load_split

EXP09 = Path(os.environ.get("EXP09_ROOT", "./data/Exp_09"))

def test_build_splits_yields_deterministic_5_5_15_per_run(tmp_path):
    splits_file = tmp_path / "splits.json"
    build_splits(exp09_root=EXP09, out=splits_file, seed=42, per_run=25)
    data = json.loads(splits_file.read_text())
    assert set(data.keys()) == {"train", "val", "test"}
    n_runs = len({e["run_id"] for e in data["test"]})
    assert n_runs >= 1
    # 5 test + 5 val + 15 train per run
    assert len(data["test"]) == 5 * n_runs
    assert len(data["val"]) == 5 * n_runs
    assert len(data["train"]) == 15 * n_runs
    # density-regime present, but NOT used for sampling - just for analysis
    assert all("density_regime" in e for e in data["test"])
    test_runs = Counter(e["run_id"] for e in data["test"])
    assert all(v == 5 for v in test_runs.values())

def test_build_splits_is_deterministic_across_runs(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    build_splits(exp09_root=EXP09, out=a, seed=42, per_run=25)
    build_splits(exp09_root=EXP09, out=b, seed=42, per_run=25)
    assert a.read_text() == b.read_text()

def test_load_split_yields_image_paths_that_exist(tmp_path):
    splits_file = tmp_path / "splits.json"
    build_splits(exp09_root=EXP09, out=splits_file, seed=42, per_run=25)
    for entry in load_split(splits_file, "test"):
        assert Path(entry.image_path).exists(), f"missing: {entry.image_path}"
