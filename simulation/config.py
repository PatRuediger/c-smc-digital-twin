import math
from mathutils import Vector


class SimulationConfig:
    """Loads all parameters from a JSON file."""

    def __init__(self, config_data, seed_override):
        """
        Initializes the configuration from a dictionary (loaded from JSON).
        :param config_data: The loaded JSON data dictionary.
        :param seed_override: The seed passed from the runner script.
        """
        sim_params = config_data.get("simulation_parameters", {})
        density_params = config_data.get("density_calculation", {})
        output_params = config_data.get("output_paths", {})

        # --- SIMULATION PARAMETERS ---
        self.number_of_strips = sim_params.get("number_of_strips", 200)
        self.spawn_interval = sim_params.get("spawn_interval", 3)
        self.simulation_frames = sim_params.get("simulation_frames", 800)
        self.belt_speed = sim_params.get("belt_speed", -0.01)

        # --- DENSITY CALCULATION ---
        self.measurement_frames = density_params.get("measurement_frames", [250, 300])
        self.measurement_box_center = tuple(density_params.get("measurement_box_center", [-6.0, 0.0, 0.1]))
        self.measurement_box_dims = tuple(density_params.get("measurement_box_dims", [15.0, 15.0, 0.2]))

        # --- Output paths ---
        self.render_output_path = output_params.get("render_output_path", "/tmp/blender_renders")
        self.db_output_path = output_params.get("db_output_path", "/tmp/simulation_results.db")

        # --- RANDOM SEED (always overwritten from outside) ---
        self.random_seed = seed_override

        # --- Fixed parameters (loaded from config.json "fixed_parameters" section) ---
        fixed = config_data.get("fixed_parameters", {})
        self.outlet_y_range = tuple(fixed.get("outlet_y_range", [-6.0, 6.0]))
        self.outlet_x_pos = fixed.get("outlet_x_pos", 2.0)
        self.outlet_z_pos = fixed.get("outlet_z_pos", 1.0)
        self.base_plate_dims = tuple(fixed.get("base_plate_dims", [20.0, 20.0]))
        self.belt_start_x = fixed.get("belt_start_x", 20.0)
        _delay_factor = fixed.get("belt_stop_delay_frames_factor", 30)
        self.belt_stop_delay_frames = _delay_factor * self.spawn_interval
        self.spawn_clearance_radius = fixed.get("spawn_clearance_radius", 0.8)
        self.spawn_height_offset = fixed.get("spawn_height_offset", 0.1)
        self.storage_x_offset_factor = fixed.get("storage_x_offset_factor", 0.01)
        self.storage_z_offset_factor = fixed.get("storage_z_offset_factor", 0.01)
        self.strip_width_range = tuple(fixed.get("strip_width_range", [0.3, 0.3]))
        self.strip_length_range = tuple(fixed.get("strip_length_range", [1.0, 1.1]))
        self.strip_height_range = tuple(fixed.get("strip_height_range", [0.05, 0.05]))
        self.max_rotation_degrees = fixed.get("max_rotation_degrees", 5.0)
        self.max_tilt_degrees = fixed.get("max_tilt_degrees", 10.0)

        print("\n--- Configuration successfully loaded ---")
        print(f"  Seed: {self.random_seed}")
        print(f"  Number of strips: {self.number_of_strips}")
        print(f"  Measurement frames: {self.measurement_frames}")
        print("-------------------------------------\n")


class StripData:
    """Stores pre-generated random data for a single strip."""

    def __init__(self, strip_id, planned_spawn_frame, width, length, height,
                 y_position, rotation_euler):
        self.strip_id = strip_id
        self.planned_spawn_frame = planned_spawn_frame
        self.actual_spawn_frame = None
        self.width = width
        self.length = length
        self.height = height
        self.y_position = y_position
        self.rotation_euler = rotation_euler
        self.half_extents = Vector((length / 2, width / 2, height / 2))
