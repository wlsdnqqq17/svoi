import bpy
import os
import sys
import math
import numpy as np
from mathutils import Vector, Euler

folder_name = sys.argv[6]
input_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", folder_name)
output_path = os.path.join("/Users/jinwoo/Documents/work/svoi/out", folder_name)

img_width = int(sys.argv[4])
img_height = int(sys.argv[5])
insertion_points = Vector([float(x) for x in sys.argv[1:4]])
rx, ry, rz = [float(x) for x in sys.argv[7:10]]
# Load camera intrinsics and pose
K = np.load(os.path.join(input_path, "K.npy"))
fx = K[0][0]
fy = K[1][1]
cx = K[0][2]
cy = K[1][2]
c2w = np.load(os.path.join(input_path, "c2w.npy"))

# Clear all objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

# Add a camera
cam_location = Vector(c2w[:3, 3])
look_dir = Vector((1, 0, 0))
target = cam_location + look_dir

cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = cam_location

cam_obj.rotation_mode = 'QUATERNION'
direction = (target - cam_location).normalized()
cam_obj.rotation_quaternion = direction.to_track_quat('-Z', 'Y')

bpy.context.scene.camera = cam_obj
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 128
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.device = 'GPU'

cam = bpy.context.scene.camera.data
cam.type = 'PERSP'
cam.lens = fx * 36 / img_width
cam.sensor_width = 36
cam.shift_x = -(cx - img_width / 2) / img_width
cam.shift_y = (cx - img_width / 2) / img_width
cam.clip_start = 0.01
print(f"카메라 추가됨: 위치={cam_location}, 방향={look_dir}")

# Import environment map
world = bpy.context.scene.world
nodes = world.node_tree.nodes
for node in nodes:
    nodes.remove(node)
tex_coord = nodes.new(type='ShaderNodeTexCoord')
mapping = nodes.new(type='ShaderNodeMapping')
env_tex = nodes.new(type='ShaderNodeTexEnvironment')
background = nodes.new(type='ShaderNodeBackground')
output = nodes.new(type='ShaderNodeOutputWorld')
tex_coord.location = (-800, 0)
mapping.location = (-600, 0)
env_tex.location = (-400, 0)
background.location = (-200, 0)
output.location = (0, 0)
mapping.inputs['Rotation'].default_value = (0, 0, math.radians(180))

links = world.node_tree.links
links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
links.new(env_tex.outputs['Color'], background.inputs['Color'])
links.new(background.outputs['Background'], output.inputs['Surface'])

# Load environment image
image_path = os.path.join(output_path, "envmap.hdr")
global_image_path = os.path.join(input_path, "global.hdr")
image = bpy.data.images.load(image_path)
global_image = bpy.data.images.load(global_image_path)
env_tex.image = image
env_tex.image.colorspace_settings.name = 'Non-Color'
image.pack()
global_image.pack()
bpy.context.scene.render.film_transparent = True

# Insert an object
insert_object = False
if insert_object:
    obj_path = os.path.join(input_path, "obj/scene.gltf")
    bpy.ops.import_scene.gltf(filepath=obj_path)
    selected_objects = bpy.context.selected_objects
    if not selected_objects:
        raise RuntimeError("No objects found in the imported GLTF file.")
    imported_obj = selected_objects[0]
    imported_obj.location = insertion_points
    imported_obj.scale = Vector((0.05, 0.05, 0.05)) 
else:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.052)
    bpy.ops.object.shade_smooth()

    sphere = bpy.context.active_object
    sphere.location = insertion_points
    mat = bpy.data.materials.new(name="SphereMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')

    bsdf.inputs['Metallic'].default_value = 1.0
    bsdf.inputs['Roughness'].default_value = 0.194
    bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    sphere.data.materials.append(mat)

# Insert Proxy plane
bpy.ops.mesh.primitive_plane_add(size=0.2)
plane = bpy.context.active_object
plane.location = insertion_points
plane.rotation_mode = 'XYZ'
plane.rotation_euler = Euler((
    math.radians(rx),
    math.radians(ry),
    math.radians(rz)
), 'XYZ')
plane.is_shadow_catcher = True
plane.visible_glossy = False

R = Euler((math.radians(rx), math.radians(ry), math.radians(rz)), 'XYZ').to_matrix()
n_world = (R @ Vector((0, 0, 1))).normalized()
radius = sphere.dimensions.x * 0.5
sphere.location = plane.location - n_world * radius

# Render 
scene = bpy.context.scene
scene.render.resolution_x = img_width
scene.render.resolution_y = img_height
scene.render.resolution_percentage = 100

scene.use_nodes = True
tree = scene.node_tree
nodes = tree.nodes
links = tree.links
for node in nodes:
    nodes.remove(node)
render_layers = nodes.new(type='CompositorNodeRLayers')
image_node = nodes.new(type='CompositorNodeImage')
alpha_over = nodes.new(type='CompositorNodeAlphaOver')
composite = nodes.new(type='CompositorNodeComposite')
scale_node = nodes.new(type='CompositorNodeScale')

img_path = os.path.join(input_path, "input.jpg")
image_node.image = bpy.data.images.load(img_path)
scale_node.space = 'RELATIVE'
scale_node.inputs[1].default_value = 1
scale_node.inputs[2].default_value = 1 

render_layers.location = (-400, 100)
image_node.location = (-600, -100)
scale_node.location = (-400, -100)
alpha_over.location = (0, 0)
composite.location = (200, 0)

links.new(image_node.outputs['Image'], scale_node.inputs[0])
links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
links.new(alpha_over.outputs['Image'], composite.inputs['Image']) 


# Save Blender file
blend_path = os.path.join(output_path, "insert_object2.blend")
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

print(f"Scene saved to {blend_path}")


# Render result
output1_path = os.path.join(output_path, "result.png")
scene.view_settings.view_transform = 'Standard'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = output1_path
bpy.ops.render.render(write_still=True)
print(f"Result 1 saved to {output_path}")

env_tex.image = global_image
output2_path = os.path.join(output_path, "result2.png")
scene.render.filepath = output2_path
bpy.ops.render.render(write_still=True)
print(f"Result 2 saved to {output_path}")