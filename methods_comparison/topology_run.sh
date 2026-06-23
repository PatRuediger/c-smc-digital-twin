#!/usr/bin/env bash
# topology_run.sh -- run TTK pipeline over one split of splits.json
#
# Usage:
#   bash methods_comparison/topology_run.sh <splits.json> <split_name> <persistence_threshold>
#
# Example:
#   bash methods_comparison/topology_run.sh methods_comparison/splits.json val 6.0
#
# Each frame is cached; frames with existing output are skipped.
# Expected wallclock: ~10-15s per frame (no OpenMP, single-threaded TTK build).
#   val split  (100 frames): ~25-40 min
#   test split (100 frames): ~25-40 min

set -euo pipefail

SPLITS=${1:?usage: topology_run.sh splits.json split_name persistence_threshold}
SPLIT=${2:?usage: topology_run.sh splits.json split_name persistence_threshold}
THR=${3:?usage: topology_run.sh splits.json split_name persistence_threshold}

PVPY=/Applications/ParaView-6.0.1.app/Contents/bin/pvpython
SCRIPT=methods_comparison/topology_ttk.py
CACHE=methods_comparison/cache/topology
mkdir -p "$CACHE"

python3 -c "
import json, sys
from pathlib import Path
data = json.load(open('$SPLITS'))
entries = data.get('$SPLIT')
if entries is None:
    print(f'ERROR: split \"$SPLIT\" not found in $SPLITS', file=sys.stderr)
    sys.exit(1)
for e in entries:
    out = Path('$CACHE') / f\"{e['run_id']}_{e['frame_id']}.json\"
    print(e['image_path'], out, sep='\t')
" | while IFS=$'\t' read -r img out; do
    if [[ -f "$out" ]]; then
        echo "SKIP (cached): $out"
        continue
    fi
    echo "PROCESS: $img"
    "$PVPY" "$SCRIPT" --input "$img" --persistence "$THR" --out "$out"
done

echo "Done. Results in $CACHE/"
