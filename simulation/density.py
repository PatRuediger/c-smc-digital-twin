import bpy
import bmesh
import math
from mathutils import Vector


class DensityMixin:
    """3D volumetric density and 2D shadow area density calculations."""

    def calculate_subsection_density(self):
        """
        Calculates the volume density of strips by applying a boolean modifier
        and correctly accounting for the object's scale.
        """
        print("\n--- Starting Volume Density Calculation (Corrected Method) ---")

        depsgraph = bpy.context.evaluated_depsgraph_get()
        print(f"Analyzing scene at frame: {bpy.context.scene.frame_current}")

        if "MeasurementBox" in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects["MeasurementBox"].select_set(True)
            bpy.ops.object.delete()

        bpy.ops.mesh.primitive_cube_add(
            location=self.config.measurement_box_center,
            scale=(self.config.measurement_box_dims[0] / 2,
                   self.config.measurement_box_dims[1] / 2,
                   self.config.measurement_box_dims[2] / 2)
        )
        measurement_box = bpy.context.active_object
        measurement_box.name = "MeasurementBox"
        measurement_box.display_type = 'WIRE'
        measurement_box.hide_render = True
        bpy.context.view_layer.objects.active = measurement_box
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        box_volume = (self.config.measurement_box_dims[0] *
                      self.config.measurement_box_dims[1] *
                      self.config.measurement_box_dims[2])
        if box_volume == 0:
            print("Error: Measurement box has zero volume.")
            return 0, 0

        box_matrix = measurement_box.matrix_world
        box_corners = [box_matrix @ v.co for v in measurement_box.data.vertices]
        box_min = [min(c[i] for c in box_corners) for i in range(3)]
        box_max = [max(c[i] for c in box_corners) for i in range(3)]

        total_strips_volume_in_box = 0.0
        strips_inside_box_count = 0
        all_strips = [obj for obj in bpy.context.scene.objects if obj.name.startswith("Strip_")]

        if not all_strips:
            print("No strip objects found in the scene.")
            density = 0.0
        else:
            print(f"Found {len(all_strips)} strips to analyze.")
            for strip in all_strips:
                strip_eval = strip.evaluated_get(depsgraph)
                strip_matrix = strip_eval.matrix_world

                strip_bb_corners = [strip_matrix @ Vector(v) for v in strip_eval.bound_box]
                strip_min = [min(c[i] for c in strip_bb_corners) for i in range(3)]
                strip_max = [max(c[i] for c in strip_bb_corners) for i in range(3)]

                x_overlap = strip_min[0] < box_max[0] and strip_max[0] > box_min[0]
                y_overlap = strip_min[1] < box_max[1] and strip_max[1] > box_min[1]
                z_overlap = strip_min[2] < box_max[2] and strip_max[2] > box_min[2]

                if not (x_overlap and y_overlap and z_overlap):
                    continue

                mod = strip.modifiers.new(name='IntersectionBoolean', type='BOOLEAN')
                mod.operation = 'INTERSECT'
                mod.object = measurement_box
                mod.solver = 'EXACT'

                intersected_obj = strip.evaluated_get(depsgraph)
                temp_mesh = intersected_obj.to_mesh()
                strip.modifiers.remove(mod)

                if not temp_mesh.vertices:
                    bpy.data.meshes.remove(temp_mesh)
                    continue

                bm = bmesh.new()
                bm.from_mesh(temp_mesh)

                base_volume = abs(bm.calc_volume(signed=True))
                bm.free()

                s = intersected_obj.scale
                actual_volume = base_volume * s.x * s.y * s.z

                if actual_volume > 1e-9:
                    total_strips_volume_in_box += actual_volume
                    strips_inside_box_count += 1

            density = (total_strips_volume_in_box / box_volume) if box_volume > 0 else 0

        mean_strip_volume = (sum(self.config.strip_width_range) / len(self.config.strip_width_range) *
                             sum(self.config.strip_length_range) / len(self.config.strip_length_range) *
                             sum(self.config.strip_height_range) / len(self.config.strip_height_range))
        mean_strip_area = (sum(self.config.strip_width_range) / len(self.config.strip_width_range) *
                           sum(self.config.strip_length_range) / len(self.config.strip_length_range))

        print("\n" + "=" * 40)
        print("--- Density Calculation Report ---")
        print(f"Total Measurement Box Volume: {box_volume:.4f} cubic meters")
        print(f"Total Volume of Strips Inside Box: {total_strips_volume_in_box:.4f} cubic meters")
        print(f"Number of Strips Inside Box: {strips_inside_box_count}")
        print(f"Mean Volume of {strips_inside_box_count} strips: {strips_inside_box_count * mean_strip_volume:.4f} cubic meters")
        print(f"Mean Area of {strips_inside_box_count} strips: {strips_inside_box_count * mean_strip_area:.4f} m²")
        print(f"Calculated Volumetric Density: {density:.4f} (or {density * 100:.2f}%)")
        print("=" * 40)
        print("The wireframe 'MeasurementBox' has been left in the scene for inspection.")

        return density, strips_inside_box_count

    def calculate_subsection_density2D(self, grid_resolution=1000):
        """
        Calculates 2D density using a high-performance BVH Tree.
        :param grid_resolution: The number of sample points along each axis.
        """
        print("\n--- Starting 2D Area Density Calculation (High-Performance BVH Method) ---")

        depsgraph = bpy.context.evaluated_depsgraph_get()
        print(f"Analyzing scene at frame: {bpy.context.scene.frame_current}")

        box_dims = self.config.measurement_box_dims
        box_center = self.config.measurement_box_center
        box_area = box_dims[0] * box_dims[1]

        if box_area == 0:
            print("Error: Measurement box has zero area.")
            return 0

        box_min_x = box_center[0] - box_dims[0] / 2
        box_max_x = box_center[0] + box_dims[0] / 2
        box_min_y = box_center[1] - box_dims[1] / 2
        box_max_y = box_center[1] + box_dims[1] / 2

        all_strips = [obj for obj in bpy.context.scene.objects if obj.name.startswith("Strip_")]
        candidate_strips = []

        for strip in all_strips:
            strip_eval = strip.evaluated_get(depsgraph)
            strip_matrix = strip_eval.matrix_world
            strip_bb_corners = [strip_matrix @ Vector(v) for v in strip_eval.bound_box]
            strip_min = [min(c[i] for c in strip_bb_corners) for i in range(3)]
            strip_max = [max(c[i] for c in strip_bb_corners) for i in range(3)]
            x_overlap = strip_min[0] < box_max_x and strip_max[0] > box_min_x
            y_overlap = strip_min[1] < box_max_y and strip_max[1] > box_min_y
            z_overlap = (strip_min[2] < (box_center[2] + box_dims[2] / 2) and
                         strip_max[2] > (box_center[2] - box_dims[2] / 2))
            if x_overlap and y_overlap and z_overlap:
                candidate_strips.append(strip_eval)

        if not candidate_strips:
            print("No relevant strips found. Density is 0%.")
            return 0
        print(f"Found {len(candidate_strips)} strips for 2D density analysis.")

        print("Building combined shadow mesh and BVH Tree...")
        combined_bm = bmesh.new()
        for strip_eval in candidate_strips:
            temp_mesh = strip_eval.to_mesh()
            temp_mesh.transform(strip_eval.matrix_world)
            combined_bm.from_mesh(temp_mesh)

        for v in combined_bm.verts:
            v.co.z = 0.0

        from mathutils.bvhtree import BVHTree
        bvh = BVHTree.FromBMesh(combined_bm)
        combined_bm.free()

        print(f"Sampling on a {grid_resolution}x{grid_resolution} grid...")
        covered_cells = 0
        z_axis = Vector((0.0, 0.0, 1.0))

        for i in range(grid_resolution):
            px = box_min_x + (i + 0.5) * (box_dims[0] / grid_resolution)
            if (i % 50 == 0) or (i == grid_resolution - 1):
                print(f"Sampling progress: {i + 1}/{grid_resolution}", end="\r")

            for j in range(grid_resolution):
                py = box_min_y + (j + 0.5) * (box_dims[1] / grid_resolution)
                ray_origin = Vector((px, py, -1.0))
                hit, _, _, _ = bvh.ray_cast(ray_origin, z_axis)
                if hit is not None:
                    covered_cells += 1

        print("\nGrid sampling complete.                            ")

        cell_area = (box_dims[0] / grid_resolution) * (box_dims[1] / grid_resolution)
        total_shadow_area = covered_cells * cell_area
        density = total_shadow_area / box_area if box_area > 0 else 0

        mean_strip_width = sum(self.config.strip_width_range) / 2
        mean_strip_length = sum(self.config.strip_length_range) / 2
        mean_strip_area = mean_strip_width * mean_strip_length

        print("\n" + "=" * 50)
        print("--- 2D Density Calculation Report ---")
        print(f"Total Measurement Area (XY Plane): {box_area:.4f} m²")
        print(f"Total Projected Shadow Area: {total_shadow_area:.4f} m²")
        print(f"Calculated 2D Area Density: {density:.4f} (or {density * 100:.2f}%)")
        print("=" * 50)

        return density
