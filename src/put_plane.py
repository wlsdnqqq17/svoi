import bpy
import os
import random
from mathutils import Vector

tex_dir = "../dataset/abstract_1-4K" 
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

# 노드 위치 조정
tex_coord.location = (-800, 0)
mapping.location = (-600, 0)
tex_image_color.location = (-400, 200)
tex_image_rough.location = (-400, 0)
tex_image_normal.location = (-400, -200)
normal_map.location = (-200, -200)
bsdf.location = (0, 0)
output.location = (200, 0)

# 연결
links.new(tex_image_color.outputs['Color'], bsdf.inputs['Base Color'])
links.new(tex_image_rough.outputs['Color'], bsdf.inputs['Roughness'])
links.new(tex_image_normal.outputs['Color'], normal_map.inputs['Color'])
links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# 머티리얼 Plane에 적용
if len(plane.data.materials) == 0:
    plane.data.materials.append(mat)
else:
    plane.data.materials[0] = mat


for image in bpy.data.images:
    if not image.packed_file:
        image.pack()

glb_paths = [
    "../dataset/1.glb",
    "../dataset/2.glb",
    "../dataset/3.glb"
]

TARGET_MAX_DIM = 1.0
PADDING = 0.5

for i, glb_path in enumerate(glb_paths):
    # 1) GLB 임포트
    bpy.ops.import_scene.gltf(filepath=glb_path)
    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']



blend_path = os.path.join(os.path.dirname(bpy.data.filepath), "plane.blend")
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)