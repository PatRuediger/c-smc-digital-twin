import bpy

from simulation.scene import SceneMixin
from simulation.physics import PhysicsMixin
from simulation.polarization import PolarizationMixin
from simulation.density import DensityMixin
from simulation.annotation import AnnotationMixin
from simulation.rendering import RenderingMixin


class StripDroppingSimulation(
    SceneMixin,
    PhysicsMixin,
    PolarizationMixin,
    DensityMixin,
    AnnotationMixin,
    RenderingMixin,
):
    """Orchestrates the entire strip dropping simulation.

    Inherits domain-specific methods from focused mixin modules:
      - SceneMixin:       scene setup, belt, spawn zone, strip creation
      - PhysicsMixin:     pre-generation of strip data, collision-free spawn scheduling
      - PolarizationMixin: shader node groups, material assignment, angle computation
      - DensityMixin:     3D volume density and 2D shadow density
      - AnnotationMixin:  YOLO OBB annotation generation
      - RenderingMixin:   polarization series rendering, AoLP computation, DB write
    """

    TEMPLATE_STRIP_NAME = "TemplateStrip"
    TEMPLATE_MESH_NAME = "TemplateStripMesh"
    BELT_NAME = "MovingBelt"
    BELT_MATERIAL_NAME = "BeltMaterial"
    STRIP_MATERIAL_NAME = "StripMaterial"
    SPAWN_ZONE_NAME = "SpawnZone"

    def __init__(self, config):
        self.config = config
        self.strip_data_list = []
        self.spawn_schedule = {}

    def run_setup(self):
        """Phase 1 of the pipeline: scene setup, strip pre-generation, spawn scheduling."""
        print("\n" + "=" * 60)
        print("STARTING COLLISION-FREE STRIP SIMULATION SETUP")
        print("=" * 60 + "\n")

        self.clear_scene()

        bpy.context.scene.frame_end = self.config.simulation_frames
        if bpy.context.scene.rigidbody_world:
            bpy.context.scene.rigidbody_world.point_cache.frame_end = self.config.simulation_frames

        self.pre_generate_strip_data()
        self.calculate_spawn_schedule()
        self.create_spawn_zone_indicator()
        self._create_animated_belt()
        self.create_all_strips_with_smart_spawning()

        bpy.context.scene.frame_set(1)
        print("\nSETUP COMPLETE.")

    def run_full_pipeline(self):
        """Runs the entire process: setup → physics bake → analysis."""
        self.run_setup()

        print("\n--- Starting physics baking ---")
        bpy.ops.ptcache.bake_all(bake=True)
        print("--- Physics baking completed ---\n")

        self.run_analysis_and_save()


def run_full_pipeline(config):
    """Module-level entry point used by the thin CLI script."""
    sim = StripDroppingSimulation(config)
    sim.run_full_pipeline()
