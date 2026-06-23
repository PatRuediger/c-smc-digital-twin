"""Wendet die geplante TTK-Pipeline auf 6 Sample-Frames an und vergleicht
'count' (Morse-Smale ascending-manifold-count) mit GT.

PASS: Pearson r >= 0.5 ueber 6 Samples (3 Density-Regime x 2) -> Topologie als
      Standalone-Counter ist plausibel.
WARN: 0.2 <= r < 0.5 -> User-Decision: standalone bleiben oder auf Sublevel-Set-
      Persistence-Counter wechseln. In non-interactive mode exits 0 with note.
FAIL: r < 0.2 -> Topologie-Methode wird auf Feature-Extractor (fuer Late-Fusion)
      umgewidmet. Plan Task 4 + 5 + 10 angepasst.
SKIP (exit 0): splits.json or topology_ttk.py missing; re-run after Task 1 + Task 4.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PVPYTHON = "/Applications/ParaView-6.0.1.app/Contents/bin/pvpython"
SCRIPT = "methods_comparison/topology_ttk.py"
INTERACTIVE = os.environ.get("PREFLIGHT_INTERACTIVE", "0") == "1"


def main() -> int:
    splits_file = Path("methods_comparison/splits.json")
    if not splits_file.exists():
        print("SKIP: splits.json missing. Run Task 1 first or temp-build a 6-sample set.")
        print("Re-run this step after Task 1 completes.")
        return 0

    splits = json.loads(splits_file.read_text())
    test = splits["test"]
    # 2 frames per regime
    by_regime: dict[str, list[dict]] = {}
    for e in test:
        by_regime.setdefault(e["density_regime"], []).append(e)
    samples = []
    for r in ("low", "medium", "high"):
        if r in by_regime:
            samples.extend(by_regime[r][:2])
    if len(samples) < 6:
        print(f"SKIP: only {len(samples)} samples available (need 6 across 3 regimes)")
        return 0

    if not Path(SCRIPT).exists():
        print(f"SKIP: {SCRIPT} not yet implemented (will be in Task 4).")
        print("Re-run this preflight step after Task 4.4 (TTK pvpython script exists).")
        return 0

    out_dir = Path("methods_comparison/preflight/ttk_pilot_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = []
    gt = []
    for e in samples:
        out = out_dir / f"{e['run_id']}_{e['frame_id']}.json"
        if not out.exists():
            res = subprocess.run(
                [PVPYTHON, SCRIPT, "--input", e["image_path"],
                 "--persistence", "6.0", "--out", str(out)],
                capture_output=True, text=True, timeout=120,
            )
            if res.returncode != 0:
                print(f"FAIL: pvpython errored on {e['image_path']}: {res.stderr[:200]}")
                return 1
        payload = json.loads(out.read_text())
        pred.append(payload["count"])
        gt.append(e["gt_count"])

    pred_arr = np.array(pred)
    gt_arr = np.array(gt)
    if np.std(pred_arr) == 0 or np.std(gt_arr) == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(pred_arr, gt_arr)[0, 1])

    print(f"GT:   {gt_arr.tolist()}")
    print(f"TTK:  {pred_arr.tolist()}")
    print(f"Pearson r = {r:.3f}  (n={len(pred_arr)})")

    decision = Path("methods_comparison/paper_method_decision.md")
    existing = decision.read_text() if decision.exists() else ""

    if r >= 0.5:
        verdict = "PASS"
        existing += f"\n\n## TTK Strategy (Step 0.6, 2026-05-07)\nStandalone counter, Pearson r = {r:.3f}.\n"
        ret = 0
    elif r >= 0.2:
        verdict = "WARN"
        if not INTERACTIVE:
            print(f"WARN: r={r:.2f} borderline. Non-interactive mode: defaulting to standalone-counter.")
            print("Set PREFLIGHT_INTERACTIVE=1 and re-run to decide interactively.")
            existing += (
                f"\n\n## TTK Strategy (Step 0.6, 2026-05-07)\n"
                f"Standalone counter (borderline, r={r:.3f}). Non-interactive default kept. Review manually.\n"
            )
            ret = 0
        else:
            ans = input(f"r={r:.2f} borderline. Keep TTK as standalone counter? [y/N] ").strip().lower()
            if ans == "y":
                existing += (
                    f"\n\n## TTK Strategy (Step 0.6, 2026-05-07)\n"
                    f"Standalone counter (borderline), Pearson r = {r:.3f}. Discussion needs hedge.\n"
                )
                ret = 0
            else:
                existing += (
                    f"\n\n## TTK Strategy (Step 0.6, 2026-05-07)\n"
                    f"Feature-extractor only (Late-Fusion), Pearson r = {r:.3f}. Task 4 to drop standalone 'count'.\n"
                )
                ret = 0
        verdict = "WARN-decided"
    else:
        verdict = "FAIL"
        existing += (
            f"\n\n## TTK Strategy (Step 0.6, 2026-05-07)\n"
            f"Feature-extractor only (Late-Fusion). Standalone count Pearson r = {r:.3f} not viable. "
            "Task 4 + 5 + 10 angepasst.\n"
        )
        ret = 1

    decision.write_text(existing)
    print(f"{verdict}: decision recorded in {decision}")
    return ret


if __name__ == "__main__":
    sys.exit(main())
