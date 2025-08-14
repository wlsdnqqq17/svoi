import bpy
import os
from mathutils import Quaternion, Vector, Euler
import math
import sys

if len(sys.argv) != 8:
    print("Usage: python gen_fenv.py <insertion_x> <insertion_y> <insertion_z> <folder_name>")
    sys.exit(1)

folder_name = sys.argv[4]
insertion_points = [float(x) for x in sys.argv[1:4]]
rx, ry, rz = [float(x) for x in sys.argv[5:8]]
base_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", folder_name)
glb_path = os.path.join(base_path, "full_scene.glb")
gltf_path = os.path.join(base_path, "full_scene.gltf")
image_path = os.path.join(base_path, "global.hdr")


print("Insertion points:", insertion_points)
# Clear existing objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

# Load the glb file
bpy.ops.import_scene.gltf(filepath=gltf_path)

q_rot = Quaternion((0, -1, 1, 0))

# Rotate all mesh objects
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = q_rot @ obj.rotation_quaternion

world = bpy.context.scene.world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links

# Remove existing nodes
for node in nodes:
    nodes.remove(node)

# Generate new nodes for the environment map
tex_coord = nodes.new(type='ShaderNodeTexCoord')
mapping = nodes.new(type='ShaderNodeMapping')
env_tex = nodes.new(type='ShaderNodeTexEnvironment')
background = nodes.new(type='ShaderNodeBackground')
output = nodes.new(type='ShaderNodeOutputWorld')

# Set node properties
tex_coord.location = (-800, 0)
mapping.location = (-600, 0)
env_tex.location = (-400, 0)
background.location = (-200, 0)
output.location = (0, 0)

# Link the nodes
links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
links.new(env_tex.outputs['Color'], background.inputs['Color'])
links.new(background.outputs['Background'], output.inputs['Surface'])

blend_dir = os.path.dirname(bpy.data.filepath)
try:
    image = bpy.data.images.load(image_path)
    env_tex.image = image
    image.pack()
    print("Success:", image.name)
except Exception as e:
    print("Fail:", e)

env_tex.image.colorspace_settings.name = 'Non-Color'
mapping.inputs['Rotation'].default_value = (0, 0, math.radians(180))

# Add a camera
cam_location = Vector(insertion_points[:3])
R = Euler((math.radians(rx), math.radians(ry), math.radians(rz)), 'XYZ').to_matrix()
n_world = (R @ Vector((0, 0, 1))).normalized()
radius = 0.052
cam_location = cam_location - n_world * radius

look_dir = Vector((-1, 0, 0))
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
cam.type = 'PANO'
cam.panorama_type = 'EQUIRECTANGULAR'
cam.clip_start = 0.01
print(f"카메라 추가됨: 위치={cam_location}, 방향={look_dir}")

# Save the scene as a blend file
blend_path = os.path.join(os.path.dirname(bpy.data.filepath), "..", f"out/{folder_name}/making_envmap.blend")
envmap_path = os.path.join(os.path.dirname(bpy.data.filepath), "..", f"out/{folder_name}/envmap.hdr")

os.makedirs(os.path.dirname(blend_path), exist_ok=True)
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

print(f"Scene saved to {blend_path}")


scene = bpy.context.scene
scene.render.resolution_x = 1024
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100

scene.render.image_settings.file_format = 'HDR'
scene.render.image_settings.color_depth = '32'
scene.view_settings.view_transform = 'Raw'


scene.render.filepath = envmap_path
bpy.ops.render.render(write_still=True)
print(f"HDR Environment map saved to {envmap_path}")
