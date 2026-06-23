"""Smoke test for topology_ttk.py (Step 4.2 / 4.4).

Run via:
    pytest methods_comparison/tests/test_topology_ttk.py -v
"""
import json
import subprocess
from pathlib import Path

PVPYTHON = "/Applications/ParaView-6.0.1.app/Contents/bin/pvpython"
SCRIPT = "methods_comparison/topology_ttk.py"
# Use the first aolp PNG found from the Exp_09 dataset (same source as splits.json)
_SAMPLE_PNG = next(
    Path(os.environ.get("EXP09_ROOT", "./data/Exp_09")).rglob(
        "*aolp*.png"
    ),
    None,
)


def test_topology_ttk_smoke(tmp_path: Path) -> None:
    assert _SAMPLE_PNG is not None, "No aolp PNG found in Exp_09 -- check dataset path"
    out = tmp_path / "topo.json"
    res = subprocess.run(
        [PVPYTHON, SCRIPT, "--input", str(_SAMPLE_PNG), "--persistence", "6.0", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert res.returncode == 0, f"pvpython failed\nstdout={res.stdout}\nstderr={res.stderr[:2000]}"
    assert out.exists(), "output JSON was not written"
    payload = json.loads(out.read_text())
    assert "count" in payload, f"'count' missing from payload: {payload}"
    assert isinstance(payload["count"], int), f"'count' is not int: {payload['count']!r}"
    summary = payload.get("persistence_summary", {})
    for key in (
        "n_pairs",
        "max_persistence",
        "total_persistence",
        "persistence_entropy",
        "betti_dim0",
        "betti_dim1",
    ):
        assert key in summary, f"'{key}' missing from persistence_summary: {summary}"
