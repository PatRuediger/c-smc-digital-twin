import bpy
import os
import math
import numpy as np
import uuid
from datetime import datetime

from simulation.database import DatabaseManager


class RenderingMixin:
    """Camera alignment, polarization series rendering, AoLP computation, and DB write."""

    def align_camera_to_box(self):
        """
        Forces the active camera to align perfectly with the measurement_box.
        Uses orthographic projection to match the annotation coordinate system.
        """
        camera = bpy.context.scene.camera
        if not camera:
            print("ERROR: No camera found to align.")
            return

        box_center = self.config.measurement_box_center
        box_dims = self.config.measurement_box_dims

        print("Aligning camera to measurement box...")
        camera.data.type = 'ORTHO'
        camera.location.x = box_center[0]
        camera.location.y = box_center[1]
        camera.location.z = 20
        camera.rotation_euler.x = 0
        camera.rotation_euler.y = 0
        camera.rotation_euler.z = 0
        camera.data.ortho_scale = max(box_dims[0], box_dims[1])

    def run_analysis_and_save(self):
        """
        Full analysis pipeline for all measurement frames:
          - renders polarization components (0, 45, 90, 135°) into separate folders
          - computes AoLP from the four components (Blender-native, no external libs)
          - generates YOLO OBB labels
          - calculates 3D/2D densities
          - inserts a row into the SQLite DB
        Errors on individual frames are logged and the loop continues.
        """
        print("\n" + "=" * 60 + "\nSTARTING ANALYSIS, RENDERING AND DATA STORAGE\n" + "=" * 60 + "\n")

        render_root = os.path.abspath(self.config.render_output_path)
        os.makedirs(render_root, exist_ok=True)
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        print(f"Images will be saved in folder: {render_root}")

        batch_id = str(uuid.uuid4()).split('-')[0]
        run_timestamp = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                         f"_{batch_id}_seed{self.config.random_seed}")

        db_manager = DatabaseManager(self.config.db_output_path)

        try:
            db_manager.connect()
            db_manager.create_table()

            for frame_to_analyze in self.config.measurement_frames:
                try:
                    print(f"\n{'─' * 20} ANALYZING FRAME {frame_to_analyze} {'─' * 20}")
                    bpy.context.scene.frame_set(frame_to_analyze)

                    base_filename = f"{run_timestamp}_frame{frame_to_analyze:04d}"

                    # === YOLO annotations ===
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    yolo_data = self.generate_yolo_annotations_from_box(depsgraph)
                    label_path = os.path.join(render_root, f"{base_filename}.txt")
                    with open(label_path, "w") as f:
                        for line in yolo_data:
                            f.write(line + "\n")
                    print(f"YOLO labels saved: {label_path} (boxes: {len(yolo_data)})")

                    # === 3D / 2D density ===
                    density_3d, strips_inside_box_count = self.calculate_subsection_density()
                    density_2d = self.calculate_subsection_density2D()
                    print(f"Density 3D: {density_3d:.6f}, Density 2D: {density_2d:.6f}, "
                          f"Strips inside: {strips_inside_box_count}")

                    # === Polarization rendering series ===
                    self.align_camera_to_box()
                    pol_root = os.path.join(render_root, "polarization_output")
                    os.makedirs(pol_root, exist_ok=True)

                    pol_angles = [0, 45, 90, 135]
                    pol_folders = {}

                    depsgraph = bpy.context.evaluated_depsgraph_get()

                    for angle in pol_angles:
                        angle_rad = math.radians(angle)
                        folder = os.path.join(pol_root, f"polar_{angle:03d}")
                        os.makedirs(folder, exist_ok=True)
                        pol_folders[angle] = folder

                        self.update_all_material_strip_angles(depsgraph)

                        for mat in bpy.data.materials:
                            if not mat.use_nodes or mat.node_tree is None:
                                continue
                            for node in mat.node_tree.nodes:
                                if (node.type == 'GROUP' and node.node_tree and
                                        node.node_tree.name == 'PolarizationShader_PerStrip'):
                                    if 'AnalyzerAngle' in node.inputs:
                                        node.inputs['AnalyzerAngle'].default_value = angle_rad

                        pol_filename = f"{base_filename}_pol{angle:03d}.png"
                        pol_fullpath = os.path.join(folder, pol_filename)
                        bpy.context.scene.render.filepath = pol_fullpath
                        bpy.ops.render.render(write_still=True)
                        print(f"Rendered polar {angle}° → {pol_fullpath}")

                    # === Compute AoLP (Blender-native image IO) ===
                    aolp_path = None
                    try:
                        imgs_lum = []
                        first_img_size = None

                        for angle in pol_angles:
                            path = os.path.join(pol_folders[angle],
                                                f"{base_filename}_pol{angle:03d}.png")
                            if path in bpy.data.images:
                                try:
                                    bpy.data.images.remove(bpy.data.images[path])
                                except Exception:
                                    pass
                            img = bpy.data.images.load(path)
                            w, h = img.size
                            if first_img_size is None:
                                first_img_size = (w, h)
                            pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
                            lum = (0.2126 * pixels[:, :, 0] +
                                   0.7152 * pixels[:, :, 1] +
                                   0.0722 * pixels[:, :, 2])
                            imgs_lum.append(lum.copy())
                            bpy.data.images.remove(img)

                        I0, I45, I90, I135 = imgs_lum
                        eps = 1e-9
                        num = I45 - I135
                        den = I0 - I90 + eps
                        aolp = 0.5 * np.arctan2(num, den)

                        aolp_norm = (aolp + (math.pi / 2.0)) / math.pi
                        aolp_norm = np.clip(aolp_norm, 0.0, 1.0)

                        h, w = aolp_norm.shape
                        rgba = np.zeros((h, w, 4), dtype=np.float32)
                        rgba[:, :, 0] = aolp_norm
                        rgba[:, :, 1] = aolp_norm
                        rgba[:, :, 2] = aolp_norm
                        rgba[:, :, 3] = 1.0

                        aolp_img_name = f"{base_filename}_aolp"
                        aolp_img = bpy.data.images.new(aolp_img_name, width=w, height=h,
                                                       float_buffer=True)
                        aolp_img.pixels = rgba.flatten().tolist()

                        aolp_folder = os.path.join(pol_root, "aolp")
                        os.makedirs(aolp_folder, exist_ok=True)
                        aolp_path = os.path.join(aolp_folder, f"{base_filename}_aolp.png")
                        aolp_img.filepath_raw = aolp_path
                        aolp_img.file_format = 'PNG'
                        aolp_img.save()
                        print(f"Saved AoLP image: {aolp_path}")
                        bpy.data.images.remove(aolp_img)

                    except Exception as e_img:
                        print(f"Could not compute AoLP: {e_img}")

                    # === Insert result to DB ===
                    try:
                        rel_aolp_path = (os.path.relpath(aolp_path,
                                                          os.path.dirname(self.config.db_output_path))
                                         if aolp_path else None)
                        rel_bbox_path = os.path.relpath(label_path,
                                                         os.path.dirname(self.config.db_output_path))
                        db_manager.insert_result(
                            config=self.config,
                            density_3d=density_3d,
                            density_2d=density_2d,
                            strips_inside_box_count=strips_inside_box_count,
                            measurement_frame=frame_to_analyze,
                            image_path_aolp=rel_aolp_path,
                            bbox_path=rel_bbox_path
                        )
                        print("DB entry inserted for frame", frame_to_analyze)
                    except Exception as e_db:
                        print(f"Warning: DB insert failed for frame {frame_to_analyze}: {e_db}")

                except Exception as frame_e:
                    print(f"Error processing frame {frame_to_analyze}: {frame_e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print("\n✅ All measurement frames processed (loop finished).")

        except Exception as e:
            print(f"Fatal error during analysis run: {e}")

        finally:
            db_manager.close()
            print("🔒 Database connection closed.")
