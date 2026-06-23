"""
simulation/ — Blender-side package for the SMC strip dropping simulation.

Blender Python cannot import from arbitrary paths without sys.path manipulation.
This __init__.py adds the package's parent directory to sys.path so that
intra-package imports (e.g. `from simulation.config import SimulationConfig`)
work correctly when the script is executed from blender -b.
"""
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))   # .../comp_strip_generator_new/simulation
_parent = os.path.dirname(_here)                       # .../comp_strip_generator_new

if _parent not in sys.path:
    sys.path.insert(0, _parent)

from simulation.config import SimulationConfig, StripData
from simulation.database import DatabaseManager
from simulation.pipeline import StripDroppingSimulation, run_full_pipeline

__all__ = [
    "SimulationConfig",
    "StripData",
    "DatabaseManager",
    "StripDroppingSimulation",
    "run_full_pipeline",
]
