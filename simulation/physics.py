import random
import math
from mathutils import Vector

from simulation.config import StripData


class PhysicsMixin:
    """Pre-generation of strip data and collision-free spawn scheduling."""

    def pre_generate_strip_data(self):
        """
        Pre-generates all random data for strips using a fixed seed.
        This ensures deterministic behavior regardless of timeline scrubbing.
        """
        random.seed(self.config.random_seed)
        self.strip_data_list = []

        for i in range(self.config.number_of_strips):
            strip_num = i + 1
            planned_spawn_frame = 1 + (i * self.config.spawn_interval)

            width = random.uniform(*self.config.strip_width_range)
            length = random.uniform(*self.config.strip_length_range)
            height = random.uniform(*self.config.strip_height_range)
            y_pos = random.uniform(*self.config.outlet_y_range)

            max_tilt_rad = math.radians(self.config.max_tilt_degrees)
            base_rotation_y = math.radians(10)
            random_tilt_x = random.uniform(-max_tilt_rad, max_tilt_rad)
            random_tilt_y = random.uniform(-max_tilt_rad, max_tilt_rad)
            random_spin_z = random.uniform(0, 2 * math.pi)
            rotation_euler = (random_tilt_x, base_rotation_y + random_tilt_y, random_spin_z)

            strip_data = StripData(
                strip_id=strip_num,
                planned_spawn_frame=planned_spawn_frame,
                width=width,
                length=length,
                height=height,
                y_position=y_pos,
                rotation_euler=rotation_euler
            )
            self.strip_data_list.append(strip_data)

        print(f"Pre-generated data for {len(self.strip_data_list)} strips with seed {self.config.random_seed}")

    def calculate_spawn_schedule(self):
        """
        Calculates actual spawn frames for each strip, ensuring no collisions
        at spawn time. Creates a deterministic spawn schedule.
        """
        print("Calculating collision-free spawn schedule...")
        self.spawn_schedule = {}

        for strip_data in self.strip_data_list:
            earliest_spawn_frame = strip_data.planned_spawn_frame

            spawn_pos = Vector((self.config.outlet_x_pos, strip_data.y_position,
                                self.config.outlet_z_pos + self.config.spawn_height_offset))

            frame_to_check = earliest_spawn_frame
            collision_found = True
            max_delay = 50

            while collision_found and frame_to_check < earliest_spawn_frame + max_delay:
                collision_found = False

                for check_frame in range(max(1, frame_to_check - 10), frame_to_check + 1):
                    if check_frame in self.spawn_schedule:
                        for other_strip_id in self.spawn_schedule[check_frame]:
                            other_strip = next((s for s in self.strip_data_list
                                               if s.strip_id == other_strip_id), None)
                            if other_strip:
                                other_pos = Vector((self.config.outlet_x_pos,
                                                   other_strip.y_position,
                                                   self.config.outlet_z_pos + self.config.spawn_height_offset))
                                distance = (spawn_pos - other_pos).length
                                min_safe_distance = (strip_data.half_extents.length +
                                                     other_strip.half_extents.length +
                                                     self.config.spawn_clearance_radius)
                                frames_since_other_spawn = frame_to_check - check_frame
                                fall_distance = 0.5 * 9.81 * ((frames_since_other_spawn / 24.0) ** 2)

                                if distance < min_safe_distance and fall_distance < 2.0:
                                    collision_found = True
                                    break

                    if collision_found:
                        break

                if collision_found:
                    frame_to_check += 1

            strip_data.actual_spawn_frame = frame_to_check
            if frame_to_check not in self.spawn_schedule:
                self.spawn_schedule[frame_to_check] = []
            self.spawn_schedule[frame_to_check].append(strip_data.strip_id)

            if frame_to_check != strip_data.planned_spawn_frame:
                print(f"  Strip {strip_data.strip_id}: Delayed from frame "
                      f"{strip_data.planned_spawn_frame} to {frame_to_check}")

        print(f"Spawn schedule complete. Strips will spawn between frames 1 and "
              f"{max(self.spawn_schedule.keys())}")
