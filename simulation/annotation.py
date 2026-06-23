import bpy
import math
from mathutils import Vector


class AnnotationMixin:
    """YOLO OBB annotation generation from simulation frame."""

    def generate_yolo_annotations_from_box(self, depsgraph):
        """
        Calculates YOLO OBB annotations for all strips within the measurement box,
        using an orthographic (top-down) projection. Coordinates are normalized
        relative to the measurement box dimensions.

        :param depsgraph: The evaluated dependency graph for the current frame.
        :return: A list of strings in YOLO OBB format.
        """
        yolo_annotations = []

        box_center = self.config.measurement_box_center
        box_dims = self.config.measurement_box_dims

        box_min_x = box_center[0] - box_dims[0] / 2
        box_max_x = box_center[0] + box_dims[0] / 2
        box_min_y = box_center[1] - box_dims[1] / 2
        box_max_y = box_center[1] + box_dims[1] / 2
        box_min_z = box_center[2] - box_dims[2] / 2
        box_max_z = box_center[2] + box_dims[2] / 2

        box_width = box_dims[0]
        box_height = box_dims[1]

        if box_width == 0 or box_height == 0:
            return []

        def clamp(x, minimum, maximum):
            return max(minimum, min(x, maximum))

        all_strips = [obj for obj in bpy.context.scene.objects if obj.name.startswith("Strip_")]

        for strip in all_strips:
            strip_eval = strip.evaluated_get(depsgraph)
            world_corners = [strip_eval.matrix_world @ v.co for v in strip_eval.data.vertices]

            strip_min_z = min(c.z for c in world_corners)
            strip_max_z = max(c.z for c in world_corners)

            if strip_max_z < box_min_z or strip_min_z > box_max_z:
                continue

            strip_min_x_world = min(c.x for c in world_corners)
            strip_max_x_world = max(c.x for c in world_corners)
            strip_min_y_world = min(c.y for c in world_corners)
            strip_max_y_world = max(c.y for c in world_corners)

            norm_min_x = (strip_min_x_world - box_min_x) / box_width
            norm_max_x = (strip_max_x_world - box_min_x) / box_width
            norm_min_y = (strip_min_y_world - box_min_y) / box_height
            norm_max_y = (strip_max_y_world - box_min_y) / box_height

            norm_min_x = clamp(norm_min_x, 0.0, 1.0)
            norm_max_x = clamp(norm_max_x, 0.0, 1.0)
            norm_min_y = clamp(norm_min_y, 0.0, 1.0)
            norm_max_y = clamp(norm_max_y, 0.0, 1.0)

            if norm_max_x <= norm_min_x or norm_max_y <= norm_min_y:
                continue

            bb_width_aabb = norm_max_x - norm_min_x
            bb_height_aabb = norm_max_y - norm_min_y
            x_center = norm_min_x + bb_width_aabb / 2
            y_center = 1.0 - (norm_min_y + bb_height_aabb / 2)

            mat = strip_eval.matrix_world
            vec_x = mat.to_3x3() @ Vector((1, 0, 0))
            vec_y = mat.to_3x3() @ Vector((0, 1, 0))

            proj_len = math.sqrt(vec_x.x ** 2 + vec_x.y ** 2)
            proj_width = math.sqrt(vec_y.x ** 2 + vec_y.y ** 2)

            norm_len = proj_len / box_width
            norm_width = proj_width / box_height

            angle_rad = self.compute_strip_projected_angle(strip, depsgraph)
            angle_rad = -angle_rad

            while angle_rad > math.pi / 2:
                angle_rad -= math.pi
            while angle_rad < -math.pi / 2:
                angle_rad += math.pi

            class_id = 0
            yolo_annotations.append(
                f"{class_id} {x_center} {y_center} {norm_len} {norm_width} {angle_rad}"
            )

        return yolo_annotations
