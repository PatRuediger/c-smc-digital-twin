"""Inspiziert alle Exp_09 SQLite-DBs und schreibt canonical schema in _constants.py.

PASS: alle 20 DBs haben dieselbe Tabelle mit denselben Spaltennamen,
      und mindestens je eine Spalte fuer (image_path, gt_count, frame_id) ist identifizierbar.
FAIL: Schema divergiert oder erwartete Spalte fehlt -> Plan Task 1.4 muss vor Implementation
      auf gefundene Spaltennamen umgestellt werden.
"""
import re
import sqlite3
import sys
from pathlib import Path

EXP09_ROOT = Path(os.environ.get("EXP09_ROOT", "./data/Exp_09"))

# heuristic name patterns: lowercase, take first match
IMG_PATTERNS = [r".*image.*path", r".*img.*path", r".*aolp.*", r".*frame.*path"]
COUNT_PATTERNS = [r".*strip.*count.*", r".*count.*strip.*", r".*n.?strips?.*", r".*gt.*count.*"]
# Only match columns whose primary purpose is a frame identifier.
# Deliberately excludes config columns like belt_stop_delay_frames, simulation_frames.
FRAME_PATTERNS = [r"measurement_frame", r"frame_(idx|index|id)", r"frame_id", r"frame_index", r"frame_num"]


def _match(cols: list[str], patterns: list[str]) -> str | None:
    for p in patterns:
        for c in cols:
            if re.fullmatch(p, c.lower()):
                return c
    return None


def main() -> int:
    dbs = sorted(EXP09_ROOT.glob("Exp_09_Polarized_OBB_*/stripGen_results.db"))
    if not dbs:
        print(f"FAIL: no DBs under {EXP09_ROOT}")
        return 1
    print(f"found {len(dbs)} DBs")

    schemas: dict[str, set[str]] = {}
    for db in dbs:
        with sqlite3.connect(db) as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )]
            for t in tables:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
                if t not in schemas:
                    schemas[t] = set(cols)
                else:
                    schemas[t].update(cols)

    # pick the table that exists in all DBs and has count-like + image-like columns
    candidate = None
    for t, cols in schemas.items():
        col_list = list(cols)
        if _match(col_list, IMG_PATTERNS) and _match(col_list, COUNT_PATTERNS):
            candidate = (t, col_list)
            break
    if not candidate:
        print("FAIL: no table with both image-path-like and count-like columns")
        for t, c in schemas.items():
            print(f"  table={t} cols={sorted(c)}")
        return 1

    t, cols = candidate
    img_col = _match(cols, IMG_PATTERNS)
    cnt_col = _match(cols, COUNT_PATTERNS)
    fr_col = _match(cols, FRAME_PATTERNS)
    fr_col_final = fr_col if fr_col else "rowid"

    # Verify schema drift: re-open each DB and confirm chosen columns exist
    for db in dbs:
        with sqlite3.connect(db) as conn:
            actual_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
            missing = []
            if img_col and img_col not in actual_cols:
                missing.append(img_col)
            if cnt_col and cnt_col not in actual_cols:
                missing.append(cnt_col)
            if fr_col and fr_col not in actual_cols:
                missing.append(fr_col)
            if missing:
                print(f"FAIL: schema drift in {db}: missing {missing}")
                return 1

    print(f"PASS: table={t} image_col={img_col} count_col={cnt_col} frame_col={fr_col_final}")

    # write into _constants.py; pattern matches both annotation form (: str | None = None)
    # and already-written assignment form (= "...") so re-runs are idempotent.
    constants_path = Path("methods_comparison/_constants.py")
    constants = constants_path.read_text()
    constants = re.sub(r"EXP09_TABLE[: ].*", f'EXP09_TABLE = "{t}"', constants)
    constants = re.sub(r"EXP09_COL_IMAGE[: ].*", f'EXP09_COL_IMAGE = "{img_col}"', constants)
    constants = re.sub(r"EXP09_COL_GT_COUNT[: ].*", f'EXP09_COL_GT_COUNT = "{cnt_col}"', constants)
    constants = re.sub(r"EXP09_COL_FRAME_ID[: ].*", f'EXP09_COL_FRAME_ID = "{fr_col_final}"', constants)
    constants_path.write_text(constants)
    return 0


if __name__ == "__main__":
    sys.exit(main())
