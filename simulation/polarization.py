import bpy
import math
import mathutils


class PolarizationMixin:
    """Per-strip polarization shader node group, material assignment, and angle computation."""

    def create_polarization_nodegroup_per_strip(self):
        """
        Create (or return) a node group that expects:
          - AnalyzerAngle (float, radians)
          - Degree (float 0..1)
          - BaseColor (color)
          - StripAngle (float, radians)  <- per-object projected long-axis angle (camera space)
        The group outputs a shader where intensity ~ cos^2(StripAngle - AnalyzerAngle).
        """
        group_name = "PolarizationShader_PerStrip"
        if group_name in bpy.data.node_groups:
            return bpy.data.node_groups[group_name]

        ng = bpy.data.node_groups.new(group_name, 'ShaderNodeTree')
        interface = ng.interface
        interface.new_socket(name="AnalyzerAngle", in_out='INPUT', socket_type='NodeSocketFloat')
        interface.new_socket(name="Degree", in_out='INPUT', socket_type='NodeSocketFloat')
        interface.new_socket(name="BaseColor", in_out='INPUT', socket_type='NodeSocketColor')
        interface.new_socket(name="StripAngle", in_out='INPUT', socket_type='NodeSocketFloat')
        interface.new_socket(name="Shader", in_out='OUTPUT', socket_type='NodeSocketShader')

        nodes = ng.nodes
        links = ng.links
        nodes.clear()

        group_in = nodes.new('NodeGroupInput'); group_in.location = (-600, 0)
        group_out = nodes.new('NodeGroupOutput'); group_out.location = (700, 0)

        # Compute cos^2(StripAngle - AnalyzerAngle)
        sub = nodes.new('ShaderNodeMath'); sub.operation = 'SUBTRACT'; sub.location = (-200, 0)
        links.new(group_in.outputs['StripAngle'], sub.inputs[0])
        links.new(group_in.outputs['AnalyzerAngle'], sub.inputs[1])

        cosn = nodes.new('ShaderNodeMath'); cosn.operation = 'COSINE'; cosn.location = (0, 0)
        links.new(sub.outputs[0], cosn.inputs[0])

        sq = nodes.new('ShaderNodeMath'); sq.operation = 'POWER'; sq.location = (200, 0)
        sq.inputs[1].default_value = 2.0
        links.new(cosn.outputs[0], sq.inputs[0])

        # Multiply by Degree
        mult = nodes.new('ShaderNodeMath'); mult.operation = 'MULTIPLY'; mult.location = (350, 0)
        links.new(sq.outputs[0], mult.inputs[0])
        links.new(group_in.outputs['Degree'], mult.inputs[1])

        # Mix with base color
        mix = nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MULTIPLY'; mix.location = (450, -150)
        mix.inputs['Fac'].default_value = 1.0
        links.new(group_in.outputs['BaseColor'], mix.inputs['Color1'])
        links.new(mult.outputs[0], mix.inputs['Color2'])

        emission = nodes.new('ShaderNodeEmission'); emission.location = (600, 0)
        links.new(mix.outputs['Color'], emission.inputs['Color'])
        links.new(emission.outputs['Emission'], group_out.inputs['Shader'])

        return ng

    def apply_polarization_material_per_strip(self, obj, base_color=(0.8, 0.8, 0.8, 1.0), degree=1.0):
        """
        Create a unique material per object and assign it. The material contains the
        PolarizationShader_PerStrip node group whose 'StripAngle' will be updated each frame.
        """
        group = self.create_polarization_nodegroup_per_strip()
        mat_name = f"Polarized_{obj.name}"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            output = nodes.new('ShaderNodeOutputMaterial'); output.location = (400, 0)
            group_node = nodes.new('ShaderNodeGroup'); group_node.location = (0, 0)
            group_node.node_tree = group

            group_node.inputs['AnalyzerAngle'].default_value = 0.0
            group_node.inputs['Degree'].default_value = degree
            group_node.inputs['BaseColor'].default_value = base_color
            group_node.inputs['StripAngle'].default_value = 0.0

            links.new(group_node.outputs['Shader'], output.inputs['Surface'])

        obj.data.materials.clear()
        obj.data.materials.append(mat)

    def compute_strip_projected_angle(self, obj, depsgraph):
        """
        Compute the strip's long-axis angle projected into the camera plane (radians).
        Assumes the strip's long axis is local X (1,0,0). Uses evaluated depsgraph object.
        Returns angle in [-pi, pi].
        """
        cam = bpy.context.scene.camera
        if cam is None:
            return 0.0

        obj_eval = obj.evaluated_get(depsgraph)
        world_dir = (obj_eval.matrix_world.to_3x3() @ mathutils.Vector((1.0, 0.0, 0.0))).normalized()

        cam_world_to_local = cam.matrix_world.inverted()
        cam_space = (cam_world_to_local.to_3x3() @ world_dir)

        angle = math.atan2(cam_space.y, cam_space.x)
        return angle

    def update_all_material_strip_angles(self, depsgraph):
        """
        Iterate all strips and update the 'StripAngle' input in each strip's material group node
        based on the current evaluated orientation (so tumbling is reflected).
        """
        all_strips = [obj for obj in bpy.context.scene.objects if obj.name.startswith("Strip_")]
        for obj in all_strips:
            try:
                angle = self.compute_strip_projected_angle(obj, depsgraph)
                mat_name = f"Polarized_{obj.name}"
                if mat_name in bpy.data.materials:
                    mat = bpy.data.materials[mat_name]
                    if mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            if node.type == 'GROUP' and node.node_tree and \
                                    node.node_tree.name == 'PolarizationShader_PerStrip':
                                if 'StripAngle' in node.inputs:
                                    node.inputs['StripAngle'].default_value = angle
            except Exception as e:
                print(f"Warning: could not update strip angle for {obj.name}: {e}")
