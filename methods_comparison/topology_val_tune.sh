#!/usr/bin/env bash
# topology_val_tune.sh -- Step 4.7: tune persistence threshold on val split
#
# Sweeps 5 thresholds (2, 4, 6, 8, 10), computes MAE vs GT for each,
# selects the best threshold, and saves it to topology_params.json.
#
# DEFERRED: do NOT run during Task 4 implementation.
# Expected wallclock: ~1.5 hours (5 thresholds x ~18 min each).
#
# Run from repository root:
#   bash methods_comparison/topology_val_tune.sh

set -euo pipefail

SPLITS=methods_comparison/splits.json
SCRIPT_DIR=methods_comparison

best_thr=""
best_mae=999999

for THR in 2 4 6 8 10; do
    echo "=== Threshold: $THR ==="
    rm -rf methods_comparison/cache/topology
    bash "$SCRIPT_DIR/topology_run.sh" "$SPLITS" val "$THR"
    MAE=$(python3 -c "
import json
import numpy as np
from pathlib import Path

splits = json.load(open('$SPLITS'))
cache = Path('methods_comparison/cache/topology')
errs = []
for e in splits['val']:
    p = cache / f\"{e['run_id']}_{e['frame_id']}.json\"
    pred = json.load(p.open())
    errs.append(abs(pred['count'] - e['gt_count']))
print(float(np.mean(errs)))
")
    echo "THR=$THR MAE=$MAE"
    python3 -c "
best_mae = $best_mae
best_thr = '$best_thr'
mae = $MAE
thr = '$THR'
if mae < best_mae:
    import json, pathlib
    print(f'New best: thr={thr} mae={mae:.1f}')
    pathlib.Path('methods_comparison/topology_params.json').write_text(
        json.dumps({'persistence_threshold': float(thr), 'val_mae': mae}, indent=2))
"
    # Update tracking variables
    if python3 -c "import sys; sys.exit(0 if $MAE < $best_mae else 1)" 2>/dev/null; then
        best_mae=$MAE
        best_thr=$THR
    fi
done

echo "Best threshold: $best_thr (MAE=$best_mae)"
echo "Saved to methods_comparison/topology_params.json"

# CRITICAL: re-run val at best threshold so the cache reflects the optimum,
# not the last sweep iteration. topology_calibrate fits on this cache.
echo
echo "=== Re-running val at best threshold ($best_thr) for calibration ==="
rm -rf methods_comparison/cache/topology
bash "$SCRIPT_DIR/topology_run.sh" "$SPLITS" val "$best_thr"

echo
echo "Next steps (in order):"
echo "  1. test split  (~25-40 min):"
echo "     bash methods_comparison/topology_run.sh methods_comparison/splits.json test $best_thr"
echo "  2. train split (~60-90 min, REQUIRED for run_fusion.py):"
echo "     bash methods_comparison/topology_run.sh methods_comparison/splits.json train $best_thr"
