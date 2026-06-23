"""Statische Code-Analyse von csrnet/train_csrnet.py.

Prueft, ob das Training Density-Maps (Gaussian-Kernel an Centroiden) als Target
verwendet, oder ob es ein Count-Regressor ist.

PASS A: Density-Map-Pipeline nachweisbar (sucht 'gaussian_filter', 'density_map',
        'gaussian_kernel', '.h5' fuer Density-Maps).
PASS B: Count-Regression nachweisbar UND User bestaetigt explizit den Methodenname
        (interaktiv). NOTE: In non-interactive (agent) mode, this path exits with code 2
        and a message; do not call input().
FAIL:   weder noch -> Plan Task 3 muss erweitert werden um Density-Map-Generation.
"""
import os
import sys
from pathlib import Path

CSRNET_TRAIN = Path("csrnet/train_csrnet.py")
DENSITY_MARKERS = ("gaussian_filter", "density_map", "gaussian_kernel",
                   "density.h5", "density_target", "GroundTruth_density")
COUNT_MARKERS = ("nn.MSELoss", "loss_fn(pred, count)", "regression_count",
                 "label = count", "labels = counts")

# Set to True when running interactively (allows input() prompt in WARN branch).
INTERACTIVE = os.environ.get("PREFLIGHT_INTERACTIVE", "0") == "1"


def main() -> int:
    if not CSRNET_TRAIN.exists():
        print(f"FAIL: {CSRNET_TRAIN} not found")
        return 1
    src = CSRNET_TRAIN.read_text()

    has_density = any(m in src for m in DENSITY_MARKERS)
    has_count = any(m in src for m in COUNT_MARKERS)

    if has_density:
        print("PASS A: density-map pipeline present in train_csrnet.py")
        matched = [m for m in DENSITY_MARKERS if m in src]
        print(f"  matched markers: {matched}")
        return 0

    if has_count and not has_density:
        print("WARN: training looks like count-regression, not CSRNet-style density estimation.")
        print("If we keep this, the paper should call the method 'VGG-Count-Regressor', not 'CSRNet'.")
        if not INTERACTIVE:
            print("EXIT CODE 2: non-interactive mode; user must decide.")
            print("Re-run with PREFLIGHT_INTERACTIVE=1 to answer the prompt, or set manually:")
            print("  (a) keep pipeline -> create methods_comparison/paper_method_decision.md with rename note")
            print("  (b) add density-map generation -> Task 3 gets new Steps 3.0a + 3.0b")
            return 2
        ans = input("Proceed with current pipeline and rename method in paper? [y/N] ").strip().lower()
        if ans == "y":
            print("PASS B: user accepted method-rename to count-regressor.")
            Path("methods_comparison/paper_method_decision.md").write_text(
                "Method 2 is a VGG-based count regressor (no density map). Paper text uses this name.\n"
            )
            return 0
        print("FAIL: user declined; Task 3 needs density-map generation step.")
        return 1

    print("FAIL: neither density-map nor count-regression markers found.")
    print("Open csrnet/train_csrnet.py manually and confirm what target the loss uses.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
