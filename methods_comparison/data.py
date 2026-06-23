import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

from methods_comparison._constants import (
    EXP09_TABLE, EXP09_COL_IMAGE, EXP09_COL_GT_COUNT, EXP09_COL_FRAME_ID,
)

DensityRegime = Literal["low", "medium", "high"]

@dataclass(frozen=True)
class SplitEntry:
    image_path: str
    gt_count: int
    density_regime: DensityRegime
    run_id: str
    frame_id: int

def _classify_regime(count: int) -> DensityRegime:
    # Empirical tertiles over all 3300 measurements in Exp_09: p33=222, p67=374.
    # Range 0..693, so the plan's original 800/1500 thresholds collapsed everything
    # into "low". Rounded to 220/375 for reporting cleanliness.
    if count <= 220:
        return "low"
    if count <= 375:
        return "medium"
    return "high"

def _read_run(db_path: Path, exp09_root: Path) -> list[SplitEntry]:
    run_id = db_path.parent.name
    rows: list[SplitEntry] = []
    with sqlite3.connect(db_path) as conn:
        query = (
            f"SELECT {EXP09_COL_FRAME_ID}, {EXP09_COL_IMAGE}, {EXP09_COL_GT_COUNT} "
            f"FROM {EXP09_TABLE}"
        )
        for frame_id, image_path, gt_count in conn.execute(query):
            # image_path is relative to exp09_root; convert to absolute
            absolute_image_path = str(exp09_root / image_path)
            rows.append(SplitEntry(
                image_path=absolute_image_path,
                gt_count=int(gt_count),
                density_regime=_classify_regime(int(gt_count)),
                run_id=run_id,
                frame_id=int(frame_id),
            ))
    return rows

def build_splits(exp09_root: Path, out: Path, seed: int, per_run: int = 25) -> None:
    """Per-Run-Split: 5 test, 5 val, 15 train. Bei 20 Runs -> 100/100/300."""
    if per_run != 25:
        raise ValueError("per_run must be 25 (5+5+15)")
    rng = random.Random(seed)
    train, val, test = [], [], []
    for db_path in sorted(exp09_root.glob("Exp_09_Polarized_OBB_*/stripGen_results.db")):
        rows = _read_run(db_path, exp09_root)
        rng.shuffle(rows)
        sample = rows[:per_run]
        if len(sample) < per_run:
            raise RuntimeError(f"{db_path}: only {len(sample)} rows, need {per_run}")
        test.extend(sample[:5])
        val.extend(sample[5:10])
        train.extend(sample[10:25])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "train": [asdict(e) for e in train],
        "val": [asdict(e) for e in val],
        "test": [asdict(e) for e in test],
    }, indent=2))

def load_split(splits_file: Path, name: Literal["train", "val", "test"]) -> Iterator[SplitEntry]:
    data = json.loads(splits_file.read_text())
    for entry in data[name]:
        yield SplitEntry(**entry)
