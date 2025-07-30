import bpy
import os
import sys
import math
from mathutils import Vector

folder_name = sys.argv[9]
input_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", folder_name)
fspy_path = os.path.join(input_path, "plane.fspy")
output_path = os.path.join("/Users/jinwoo/Documents/work/svoi/out", folder_name)

def intersect_with_yz_plane(C, P):
    dir = P - C
    t = -C.x / dir.x
    return C + dir * t

def pixel_to_world(px, py):
    cam = bpy.context.scene.camera
    print(f"Camera: {cam.name}")
    frame_local = cam.data.view_frame(scene=bpy.context.scene)
    rt, rb, lb, lt = [cam.matrix_world @ v for v in frame_local]
    width = int(sys.argv[7])
    height = int(sys.argv[8])

    x_ratio = px / width
    top_interp = lt.lerp(rt, x_ratio)
    bottom_interp = lb.lerp(rb, x_ratio)
    y_ratio = 1 - py / height
    near_point = bottom_interp.lerp(top_interp, y_ratio)
    camera_location = cam.matrix_world.translation
    return intersect_with_yz_plane(camera_location, near_point)


# Clear existing objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

# Load the fspy file
base_dir = os.getcwd()
bpy.ops.fspy_blender.import_project(filepath=fspy_path)
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.cycles.use_denoising = True
world = bpy.context.scene.world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links

# Import environment map
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
links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
links.new(env_tex.outputs['Color'], background.inputs['Color'])
links.new(background.outputs['Background'], output.inputs['Surface'])
image_path = os.path.join(output_path, "envmap.png")
image = bpy.data.images.load(image_path)
env_tex.image = image
image.pack()
bpy.context.scene.render.film_transparent = True
print("Imported:", image.name)

# Browse insertion objects
px, py = int(sys.argv[5]), int(sys.argv[6])
x, y, z = pixel_to_world(px, py)
insertion_point = Vector((x, y, z))
obj_path = os.path.join(input_path, "obj/scene.gltf")
bpy.ops.import_scene.gltf(filepath=obj_path)


for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        obj.location = insertion_point
        print(f"Inserted object: {obj.name} at {obj.location}")
    

env_tex.image.colorspace_settings.name = 'Filmic sRGB'
mapping.inputs['Rotation'].default_value = (0, 0, math.radians(180))

# Render
scene = bpy.context.scene
scene.render.resolution_percentage = 25

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
scale_node.inputs[1].default_value = 0.25 
scale_node.inputs[2].default_value = 0.25 

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
blend_path = os.path.join(output_path, "insert_object.blend")
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

print(f"Scene saved to {blend_path}")

# Render result
output_path = os.path.join(output_path, "result.png")
scene.view_settings.view_transform = 'Standard'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)
print(f"Result map saved to {output_path}")