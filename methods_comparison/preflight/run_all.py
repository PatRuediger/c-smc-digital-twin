"""Laeuft alle Pre-Flight-Steps in Reihenfolge, schreibt preflight_report.md.

Usage:
    python3 -m methods_comparison.preflight.run_all

Steps 0.7, 0.8, 0.9 are interactive (User-Decision) and are listed as PENDING.
Run them manually after this script completes:
    PREFLIGHT_INTERACTIVE=1 python3 -m methods_comparison.preflight.ttk_pilot  (if splits.json exists)
    Then fill methods_comparison/paper_claim.md (A4 + A5) and methods_comparison/timeline.md (dates).
"""
import datetime
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable  # use the same interpreter that runs this script

STEPS = [
    ("0.1 schema",          [PYTHON, "-m", "methods_comparison.preflight.check_exp09_schema"]),
    ("0.2 measurement-box", [PYTHON, "-m", "methods_comparison.preflight.extract_measurement_box"]),
    ("0.3 csrnet-training", [PYTHON, "-m", "methods_comparison.preflight.audit_csrnet_training"]),
    ("0.4 csrnet-layers",   [PYTHON, "-m", "methods_comparison.preflight.check_csrnet_layers"]),
    ("0.5 aolp-histograms", [PYTHON, "-m", "methods_comparison.preflight.check_aolp_histograms"]),
    ("0.6 ttk-pilot",       [PYTHON, "-m", "methods_comparison.preflight.ttk_pilot"]),
    ("0.10 dotx-inspect",   [PYTHON, "-m", "methods_comparison.preflight.inspect_dotx"]),
]
INTERACTIVE = [
    ("0.7 real-frame-set", "Step 0.7 must run interactively (User-Decision A4)."),
    ("0.8 headline-claim", "Step 0.8 must run interactively (User-Decision A5)."),
    ("0.9 deadline",       "Step 0.9 must run interactively (User-Decision D1/D2)."),
]


def main() -> int:
    report_lines = [
        "# Pre-Flight Report",
        f"\nrun: {datetime.datetime.now().isoformat()}\n",
    ]
    overall = 0
    for name, cmd in STEPS:
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Exit code 2 = non-interactive WARN (treated as WARN, not FAIL for overall)
        if res.returncode == 0:
            verdict = "PASS"
        elif res.returncode == 2:
            verdict = "WARN"
        else:
            verdict = "FAIL"
            overall = 1
        output = (res.stdout.strip() + "\n" + res.stderr.strip()).strip()
        report_lines.append(f"\n## {name}: {verdict}\n```\n{output}\n```\n")
        print(f"  {name}: {verdict}")

    for name, msg in INTERACTIVE:
        report_lines.append(f"\n## {name}: PENDING\n{msg}\n")
        print(f"  {name}: PENDING")

    report_path = Path("methods_comparison/preflight_report.md")
    report_path.write_text("\n".join(report_lines))
    print(f"\noverall: {'PASS' if overall == 0 else 'FAIL'}")
    print(f"see {report_path}")
    return overall


if __name__ == "__main__":
    sys.exit(main())
