import argparse
import json
import math
import os

import bpy
import numpy as np
from mathutils import Euler, Vector
from scipy.spatial.transform import Rotation as R


def parse_arguments():
    parser = argparse.ArgumentParser(description="Blender 오브젝트 삽입 스크립트")
    parser.add_argument("folder_name", type=str, help="폴더 이름")

    return parser.parse_args()


def make_transform_matrix(pos, euler_deg):
    rot = R.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


args = parse_arguments()

folder_name = args.folder_name
input_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", folder_name)
output_path = os.path.join("/Users/jinwoo/Documents/work/svoi/out", folder_name)
os.makedirs(output_path, exist_ok=True)

dataset_path = os.path.join("/Users/jinwoo/Documents/work/svoi/dataset", folder_name)
json_path = os.path.join(dataset_path, f"{folder_name}_metadata.json")

img_path = os.path.join(input_path, "input.jpg")

if not os.path.exists(img_path):
    dataset_img_path = os.path.join(
        "/Users/jinwoo/Documents/work/svoi/dataset",
        folder_name,
        f"{folder_name}_before.png",
    )
    if os.path.exists(dataset_img_path):
        img_path = dataset_img_path
        print(f"Using dataset image: {img_path}")

image = bpy.data.images.load(img_path)
img_width, img_height = image.size

with open(json_path, "r") as f:
    metadata = json.load(f)

T_A = make_transform_matrix(metadata["camera_location"], metadata["camera_rotation"])
T_B = make_transform_matrix([0.0, 0.0, 0.0], (90.0, 0.0, -90.0))

T_o = make_transform_matrix(
    metadata["insertion_object"]["location"], metadata["insertion_object"]["rotation"]
)

T_o_prime = T_B @ np.linalg.inv(T_A) @ T_o

insertion_points = T_o_prime[:3, 3]
insertion_rotation = R.from_matrix(T_o_prime[:3, :3]).as_euler("xyz", degrees=True)
print(f"Insertion points: {insertion_points}")
print(f"Insertion rotation: {insertion_rotation}")

# Clear all objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

# Add a camera
cam_location = Vector([0.0, 0.0, 0.0])
look_dir = Vector((1, 0, 0))
target = cam_location + look_dir

cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = cam_location

cam_obj.rotation_mode = "QUATERNION"
direction = (target - cam_location).normalized()
cam_obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")

scene = bpy.context.scene
assert isinstance(scene, bpy.types.Scene)

scene.camera = cam_obj
scene.render.engine = "CYCLES"

if scene.cycles is None:
    raise RuntimeError("Cycles is not available")
scene.cycles.samples = 128
scene.cycles.use_denoising = True
scene.cycles.device = "GPU"

cam = scene.camera.data
assert isinstance(cam, bpy.types.Camera)
cam.lens = 19
cam.sensor_width = 36
cam.sensor_height = 24
cam.clip_start = 0.01
print(f"카메라 추가됨: 위치={cam_location}, 방향={look_dir}")

# Import environment map
world = scene.world
nodes = world.node_tree.nodes
nodes.clear()

tex_coord = nodes.new(type="ShaderNodeTexCoord")
mapping = nodes.new(type="ShaderNodeMapping")
env_tex = nodes.new(type="ShaderNodeTexEnvironment")
background = nodes.new(type="ShaderNodeBackground")
output = nodes.new(type="ShaderNodeOutputWorld")
tex_coord.location = (-800, 0)
mapping.location = (-600, 0)
env_tex.location = (-400, 0)
background.location = (-200, 0)
output.location = (0, 0)
mapping.inputs["Rotation"].default_value = (0, 0, 0)

links = world.node_tree.links
links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
links.new(env_tex.outputs["Color"], background.inputs["Color"])
links.new(background.outputs["Background"], output.inputs["Surface"])

# Load environment image
image_path = os.path.join(input_path, "global.exr")

env_image = bpy.data.images.load(image_path)
env_tex.image = env_image

# Set proper colorspace based on file type
env_tex.image.colorspace_settings.name = "Linear Rec.709"
env_image.pack()
bpy.context.scene.render.film_transparent = True

# Insert an object
obj_path = os.path.join(dataset_path, metadata["insertion_object"]["file"])
bpy.ops.import_scene.gltf(filepath=obj_path)
selected_objects = bpy.context.selected_objects
if not selected_objects:
    raise RuntimeError("No objects found in the imported GLTF file.")
imported_obj = selected_objects[0]
imported_obj.location = insertion_points
imported_obj.rotation_mode = "XYZ"
imported_obj.rotation_euler = Euler(
    [math.radians(angle) for angle in insertion_rotation]
)
imported_obj.scale = Vector(metadata["insertion_object"]["scale"])

# Render
scene = bpy.context.scene
assert isinstance(scene, bpy.types.Scene), "Scene is not available"

scene.render.resolution_x = img_width
scene.render.resolution_y = img_height
scene.render.resolution_percentage = 100

tree = bpy.data.node_groups.new(name="Compositing Nodetree", type="CompositorNodeTree")

scene.compositing_node_group = tree
nodes = tree.nodes
links = tree.links

# Nodes
render_layers = nodes.new(type="CompositorNodeRLayers")
render_layers.layer = scene.view_layers[0].name

image_node = nodes.new(type="CompositorNodeImage")
alpha_over = nodes.new(type="CompositorNodeAlphaOver")
group_output = nodes.new(type="NodeGroupOutput")

image_node.image = image

iface = tree.interface
iface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
tree.interface_update(bpy.context)

# Layout
image_node.location = (-600, -100)
render_layers.location = (-400, -100)
alpha_over.location = (0, 0)
group_output.location = (200, 0)

links.new(image_node.outputs["Image"], alpha_over.inputs["Background"])
links.new(render_layers.outputs["Image"], alpha_over.inputs["Foreground"])
links.new(alpha_over.outputs["Image"], group_output.inputs["Image"])

# Render result
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(output_path, "result1.png")

bpy.ops.render.render(write_still=True)

# Save Blender file
blend_path = os.path.join(output_path, "insert_object.blend")
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

print(f"Scene saved to {blend_path}")
