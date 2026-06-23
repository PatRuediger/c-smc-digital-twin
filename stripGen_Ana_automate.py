"""
stripGen_Ana_automate.py — CLI entry point for the SMC strip dropping simulation.

Usage (headless Blender):
    blender -b StripsGen_outLetSimulation_init.blend \
            -P stripGen_Ana_automate.py -- \
            --seed 42 --config config.json

All simulation logic lives in the simulation/ package.
"""
import sys
import os
import json
import bpy

# ── Blender sys.path setup ────────────────────────────────────────────────────
# Blender Python cannot import from arbitrary paths without explicit sys.path
# manipulation. We add the script's own directory so that `import simulation`
# resolves correctly regardless of how Blender was launched.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from simulation.config import SimulationConfig
from simulation.pipeline import run_full_pipeline

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Strip Blender's own argv; everything after "--" belongs to the script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    # Determine config file path (default: same folder as the .blend file)
    if bpy.data.filepath:
        script_dir = os.path.dirname(bpy.data.filepath)
    else:
        script_dir = os.getcwd()

    config_path = os.path.join(script_dir, "config.json")

    if "--config" in argv:
        try:
            idx = argv.index("--config") + 1
            if idx < len(argv):
                config_path = argv[idx]
        except ValueError:
            pass

    print(f"\n--- Loading configuration from: {os.path.abspath(config_path)} ---")

    try:
        with open(config_path, 'r') as f:
            master_config_data = json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load config.json from {config_path}: {e}")
        bpy.ops.wm.quit_blender()

    seed_override = 42
    if "--seed" in argv:
        try:
            seed_index = argv.index("--seed") + 1
            if seed_index < len(argv):
                seed_override = int(argv[seed_index])
        except (ValueError, IndexError):
            pass

    config = SimulationConfig(config_data=master_config_data, seed_override=seed_override)
    run_full_pipeline(config)

    print("\n\nFULL PIPELINE COMPLETED.")
    # bpy.ops.wm.quit_blender()  # Uncomment to auto-close Blender after each run
