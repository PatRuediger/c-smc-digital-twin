import bpy
import math
from mathutils import Vector


class SceneMixin:
    """Scene setup, belt animation, spawn zone indicator, and strip creation."""

    def clear_scene(self):
        """Clears all mesh objects and resets physics."""
        if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()

        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = self.config.simulation_frames

        if bpy.context.scene.rigidbody_world:
            bpy.ops.ptcache.free_bake_all()
            rbw_objects = bpy.context.scene.rigidbody_world.collection.objects
            while rbw_objects:
                rbw_objects.unlink(rbw_objects[0])

        for material in bpy.data.materials:
            if not material.users:
                bpy.data.materials.remove(material)

    def create_spawn_zone_indicator(self):
        """Creates a wireframe visual indicator for the spawn zone."""
        zone_width = self.config.outlet_y_range[1] - self.config.outlet_y_range[0]
        zone_height = 2.0
        zone_depth = 2.0

        bpy.ops.mesh.primitive_cube_add(
            location=(self.config.outlet_x_pos,
                      (self.config.outlet_y_range[0] + self.config.outlet_y_range[1]) / 2,
                      self.config.outlet_z_pos + self.config.spawn_height_offset),
            scale=(zone_depth / 2, zone_width / 2, zone_height / 2)
        )

        spawn_zone = bpy.context.active_object
        spawn_zone.name = self.SPAWN_ZONE_NAME
        spawn_zone.display_type = 'WIRE'
        spawn_zone.show_wire = True
        spawn_zone.hide_render = True
        spawn_zone.color = (0.2, 0.8, 0.2, 0.3)
        spawn_zone.show_in_front = False

        print(f"Created spawn zone indicator at outlet position")

    def _create_animated_belt(self):
        """Creates the base plate and animates it to act as a moving belt."""
        bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD')
        belt = bpy.context.active_object
        belt.name = self.BELT_NAME

        if self.BELT_MATERIAL_NAME not in bpy.data.materials:
            mat = bpy.data.materials.new(name=self.BELT_MATERIAL_NAME)
        else:
            mat = bpy.data.materials[self.BELT_MATERIAL_NAME]
        belt.dimensions = (self.config.base_plate_dims[0] * 5, self.config.base_plate_dims[1], 0)
        belt.data.materials.append(mat)

        bpy.ops.rigidbody.object_add(type='PASSIVE')
        belt.rigid_body.collision_shape = 'BOX'
        belt.rigid_body.collision_margin = 0.001
        belt.rigid_body.kinematic = True
        belt.rigid_body.friction = 0.9

        if self.spawn_schedule:
            last_spawn_frame = max(self.spawn_schedule.keys())
        else:
            last_spawn_frame = self.config.simulation_frames // 2

        stop_frame = last_spawn_frame
        stop_frame = max(stop_frame, self.config.simulation_frames)

        start_frame = 1
        belt.location.x = self.config.belt_start_x
        belt.keyframe_insert(data_path="location", frame=start_frame, index=0)

        movement_until_stop = self.config.belt_speed * (stop_frame - start_frame)
        belt.location.x = self.config.belt_start_x + movement_until_stop
        belt.keyframe_insert(data_path="location", frame=stop_frame, index=0)

        belt.keyframe_insert(data_path="location", frame=self.config.simulation_frames, index=0)

        if belt.animation_data and belt.animation_data.action:
            for fcurve in belt.animation_data.action.fcurves:
                if fcurve.data_path == 'location' and fcurve.array_index == 0:
                    for kf in fcurve.keyframe_points:
                        kf.interpolation = 'LINEAR'

        print(f"Belt will stop at frame {stop_frame}")

    def create_all_strips_with_smart_spawning(self):
        """
        Creates all strips at once but uses the calculated spawn schedule
        to ensure collision-free spawning.
        """
        if self.STRIP_MATERIAL_NAME not in bpy.data.materials:
            bpy.data.materials.new(name=self.STRIP_MATERIAL_NAME)

        for strip_data in self.strip_data_list:
            mesh = bpy.data.meshes.new(f"StripMesh_{strip_data.strip_id}")
            strip = bpy.data.objects.new(f"Strip_{strip_data.strip_id}", mesh)

            verts = [
                (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5),
                (0.5, 0.5, -0.5), (0.5, -0.5, -0.5),
                (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)
            ]
            faces = [
                (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)
            ]
            mesh.from_pydata(verts, [], faces)
            mesh.update()

            bpy.context.scene.collection.objects.link(strip)

            strip.dimensions = (strip_data.length, strip_data.width, strip_data.height)
            base_color = (0.8, 0.6, 0.4, 1.0)
            degree = 1.0
            self.apply_polarization_material_per_strip(strip, base_color=base_color, degree=degree)

            strip.rotation_euler = strip_data.rotation_euler

            spawn_location = Vector((
                self.config.outlet_x_pos,
                strip_data.y_position,
                self.config.outlet_z_pos + self.config.spawn_height_offset
            ))
            storage_location = Vector((
                self.config.outlet_x_pos + strip_data.strip_id * self.config.storage_x_offset_factor,
                strip_data.y_position,
                self.config.outlet_z_pos + strip_data.strip_id * self.config.storage_z_offset_factor
            ))

            strip.location = storage_location

            bpy.context.view_layer.objects.active = strip
            bpy.ops.rigidbody.object_add(type='ACTIVE')
            strip.rigid_body.collision_shape = 'BOX'
            strip.rigid_body.mass = 0.1
            strip.rigid_body.linear_damping = 0.6
            strip.rigid_body.angular_damping = 0.8
            strip.rigid_body.collision_margin = 0.001
            strip.rigid_body.use_deactivation = False
            strip.rigid_body.friction = 0.3
            strip.rigid_body.restitution = 0.1

            actual_spawn_frame = strip_data.actual_spawn_frame

            if actual_spawn_frame > 1:
                strip.location = storage_location
                strip.rigid_body.kinematic = True
                strip.keyframe_insert(data_path="location", frame=1)
                strip.rigid_body.keyframe_insert(data_path="kinematic", frame=1)

                strip.location = storage_location
                strip.rigid_body.kinematic = True
                strip.keyframe_insert(data_path="location", frame=actual_spawn_frame - 1)
                strip.rigid_body.keyframe_insert(data_path="kinematic", frame=actual_spawn_frame - 1)

            strip.location = spawn_location
            strip.rigid_body.kinematic = False
            strip.keyframe_insert(data_path="location", frame=actual_spawn_frame)
            strip.rigid_body.keyframe_insert(data_path="kinematic", frame=actual_spawn_frame)

            if strip.animation_data and strip.animation_data.action:
                for fcurve in strip.animation_data.action.fcurves:
                    if fcurve.data_path == "location":
                        for kf in fcurve.keyframe_points:
                            if kf.co[0] < actual_spawn_frame:
                                kf.interpolation = 'CONSTANT'
                            else:
                                kf.interpolation = 'BEZIER'

            print(f"Strip_{strip_data.strip_id}: Scheduled for frame {actual_spawn_frame}")
