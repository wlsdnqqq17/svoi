import bpy
import os
import random
import sys
from mathutils import Vector

if len(sys.argv) > 1:
    dir_name = sys.argv[1]
else:
    dir_name = "001"

tex_dir = f"dataset/{dir_name}/abstract_1-4K" 
name = "4K-abstract_1"

for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

color_path = os.path.join(tex_dir, f"{name}-diffuse.jpg")
roughness_path = os.path.join(tex_dir, f"{name}-specular.jpg")
normal_path = os.path.join(tex_dir, f"{name}-normal.jpg")

bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = "AbstractPlane"
plane_size = plane.dimensions.xy 


mat = bpy.data.materials.new(name="Abstract_Material")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

for node in nodes:
    nodes.remove(node)

tex_coord = nodes.new(type='ShaderNodeTexCoord')
mapping = nodes.new(type='ShaderNodeMapping')

tex_image_color = nodes.new(type='ShaderNodeTexImage')
tex_image_color.image = bpy.data.images.load(color_path)

tex_image_rough = nodes.new(type='ShaderNodeTexImage')
tex_image_rough.image = bpy.data.images.load(roughness_path)
tex_image_rough.image.colorspace_settings.name = 'Non-Color'

tex_image_normal = nodes.new(type='ShaderNodeTexImage')
tex_image_normal.image = bpy.data.images.load(normal_path)
tex_image_normal.image.colorspace_settings.name = 'Non-Color'

normal_map = nodes.new(type='ShaderNodeNormalMap')

bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
output = nodes.new(type='ShaderNodeOutputMaterial')

tex_coord.location = (-800, 0)
mapping.location = (-600, 0)
tex_image_color.location = (-400, 200)
tex_image_rough.location = (-400, 0)
tex_image_normal.location = (-400, -200)
normal_map.location = (-200, -200)
bsdf.location = (0, 0)
output.location = (200, 0)

links.new(tex_image_color.outputs['Color'], bsdf.inputs['Base Color'])
links.new(tex_image_rough.outputs['Color'], bsdf.inputs['Roughness'])
links.new(tex_image_normal.outputs['Color'], normal_map.inputs['Color'])
links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

if len(plane.data.materials) == 0:
    plane.data.materials.append(mat)
else:
    plane.data.materials[0] = mat


for image in bpy.data.images:
    if not image.packed_file:
        image.pack()

glb_paths = [
    f"dataset/{dir_name}/1.glb",
    f"dataset/{dir_name}/2.glb",
    f"dataset/{dir_name}/3.glb"
]

TARGET_MAX_DIM = 1.0
PADDING = 0.5

for i, glb_path in enumerate(glb_paths):
    existing_objects = set(bpy.data.objects)
    
    bpy.ops.import_scene.gltf(filepath=glb_path)
    
    new_objects = set(bpy.data.objects) - existing_objects
    
    parent_object = None
    for obj in new_objects:
        if obj.parent is None and (obj.type == 'EMPTY' or any(child in new_objects for child in obj.children)):
            parent_object = obj
            break
    
    if parent_object is None:
        mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
        if mesh_objects:
            parent_object = mesh_objects[0]
    
    if parent_object:
        if parent_object.type == 'EMPTY':
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            def get_all_mesh_children(obj):
                meshes = []
                if obj.type == 'MESH':
                    meshes.append(obj)
                for child in obj.children:
                    meshes.extend(get_all_mesh_children(child))
                return meshes
            
            mesh_children = get_all_mesh_children(parent_object)
            
            for mesh_obj in mesh_children:
                for vertex in mesh_obj.data.vertices:
                    world_vertex = mesh_obj.matrix_world @ vertex.co
                    min_x = min(min_x, world_vertex.x)
                    min_y = min(min_y, world_vertex.y)
                    min_z = min(min_z, world_vertex.z)
                    max_x = max(max_x, world_vertex.x)
                    max_y = max(max_y, world_vertex.y)
                    max_z = max(max_z, world_vertex.z)
            
            if min_x != float('inf'):
                total_width = max_x - min_x
                total_height = max_y - min_y
                total_depth = max_z - min_z
                current_dim = max(total_width, total_height, total_depth)
            else:
                current_dim = 1.0  # fallback
        else:
            current_dim = max(parent_object.dimensions.x, parent_object.dimensions.y, parent_object.dimensions.z)
        
        scale_factor = TARGET_MAX_DIM / current_dim if current_dim > 0 else 1.0
        
        parent_object.scale = (scale_factor, scale_factor, scale_factor)
        
        x_pos = (i - 1) * (TARGET_MAX_DIM + PADDING)
        parent_object.location = (x_pos, 0, TARGET_MAX_DIM / 2)
        
        print(f"GLB {i+1}: parent='{parent_object.name}', current_dim: {current_dim:.3f}, scale_factor: {scale_factor:.3f}, position: ({x_pos:.2f}, 0, {TARGET_MAX_DIM/2:.2f})")
    else:
        print(f"GLB {i+1}: No parent object found")

blend_path = os.path.join(f"dataset/{dir_name}", f"{dir_name}.blend")
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)