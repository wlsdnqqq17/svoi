import bpy
import os
import random
from mathutils import Vector

tex_dir = "./dataset/Asphalt013_4K-JPG" 
name = "Asphalt013_4K"

for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

color_path = os.path.join(tex_dir, f"{name}_Color.jpg")
roughness_path = os.path.join(tex_dir, f"{name}_Roughness.jpg")
normal_path = os.path.join(tex_dir, f"{name}_NormalGL.jpg")

bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = "AsphaltPlane"
plane_size = plane.dimensions.xy 


mat = bpy.data.materials.new(name="Asphalt_Material")
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
    "./dataset/1.glb",
    "./dataset/2.glb",
    "./dataset/3.glb"
]

TARGET_MAX_DIM = 1.0
PADDING = 0.5

for i, glb_path in enumerate(glb_paths):
    # 1) GLB 임포트
    bpy.ops.import_scene.gltf(filepath=glb_path)
    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not imported:
        print(f"[WARNING] No mesh found in {glb_path}")
        continue
    
    # 2) 여러 메쉬가 있으면 하나로 병합
    bpy.ops.object.select_all(action='DESELECT')
    for o in imported: o.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = f"InsertedObject_{i}"
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    current_max = max(obj.dimensions)
    factor = TARGET_MAX_DIM / current_max
    obj.scale = (factor, factor, factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    half_x = plane.dimensions.x / 2 - PADDING
    half_y = plane.dimensions.y / 2 - PADDING
    rand_x = random.uniform(-half_x, half_x)
    rand_y = random.uniform(-half_y, half_y)
    print(f"[INFO] Placing object {i} at ({rand_x}, {rand_y})")
    obj.location = (rand_x, 0, rand_y)  


    print(f"[OK] Inserted {obj.name} at origin")

blend_path = os.path.join(os.path.dirname(bpy.data.filepath), "plane.blend")
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)