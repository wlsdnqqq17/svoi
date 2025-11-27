import argparse
import os
import re
import sys

import bpy


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Blender 오브젝트 삽입 스크립트 (Ours)"
    )
    parser.add_argument("experiment_number", type=str, help="실험 번호")

    return parser.parse_args()


args = parse_arguments()

experiment_number = args.experiment_number
print(f"실험 번호: {experiment_number}")

output_path = os.path.join("/Users/jinwoo/Documents/work/svoi/out", experiment_number)
input_path = os.path.join("/Users/jinwoo/Documents/work/svoi/input", experiment_number)

# Load insert_object.blend and save as insert_object4.blend
source_blend_path = os.path.join(output_path, "insert_object.blend")
target_blend_path = os.path.join(output_path, "insert_object4.blend")
glb_path = os.path.join(input_path, "full_scene.glb")

if not os.path.exists(source_blend_path):
    print(f"Error: {source_blend_path} 파일을 찾을 수 없습니다.")
    sys.exit(1)

if not os.path.exists(glb_path):
    print(f"Error: {glb_path} 파일을 찾을 수 없습니다.")
    sys.exit(1)

# Open the source blend file
bpy.ops.wm.open_mainfile(filepath=source_blend_path)
print(f"Opened: {source_blend_path}")

# Set viewport samples to 64
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.preview_samples = 64
print("Viewport samples set to 64")

# Import full_scene.glb
bpy.ops.import_scene.gltf(filepath=glb_path)
print(f"Imported: {glb_path}")

# Apply rotation only to object named "월드"
world_obj = None
for obj in bpy.context.scene.objects:
    if "world" in obj.name:
        world_obj = obj
        break

if world_obj:
    world_obj.rotation_mode = "XYZ"
    world_obj.rotation_euler[2] = 1.5708
    world_obj.rotation_euler[1] = 3.14159
    print(f"Applied rotation to object: {world_obj.name}")
else:
    print("Warning: Object with name containing 'world' not found")

# Check materials for obj files and set emission textures
obj_objects = [
    obj
    for obj in bpy.context.scene.objects
    if ".obj" in obj.name.lower() and obj.type == "MESH"
]
material_to_image_strength = {}  # material_name -> (image_path, strength)

for obj in obj_objects:
    print(f"{obj.name}: ", end="")
    if obj.data.materials:
        material_names = [
            mat_slot.material.name if mat_slot.material else "None"
            for mat_slot in obj.material_slots
        ]
        print(", ".join(material_names))
    else:
        print("Material 없음")

# Set emission texture and strength for each material based on material name
for mat_name, mat in bpy.data.materials.items():
    if not mat.use_nodes:
        continue

    bsdf_node = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf_node:
        continue

    # Extract number from material name (e.g., Material_0 -> Image_0)
    match = re.search(r"Material_(\d+)", mat_name)
    if match:
        mat_number = match.group(1)
        # Fixed image name format: Image_{number}
        img_name = f"Image_{mat_number}"

        image = None
        if img_name in bpy.data.images:
            image = bpy.data.images[img_name]

        if image:
            # Create image texture node
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Remove existing emission texture nodes if any
            nodes_to_remove = [
                node
                for node in nodes
                if node.type == "TEX_IMAGE"
                and any(
                    link.to_node == bsdf_node
                    and link.to_socket.name == "Emission Color"
                    for link in links
                    if link.from_node == node
                )
            ]
            for node in nodes_to_remove:
                nodes.remove(node)

            # Create new image texture node
            tex_image = nodes.new(type="ShaderNodeTexImage")
            tex_image.image = image
            tex_image.location = (-400, -200)

            # Connect to Emission Color and set Emission Strength (always 1.0)
            emission_strength = 1.0

            # Connect image texture to Emission Color
            links.new(tex_image.outputs["Color"], bsdf_node.inputs["Emission Color"])
            bsdf_node.inputs["Emission Strength"].default_value = emission_strength
    else:
        print(f"  {mat_name}: Material_숫자 형식이 아닙니다")
print("=" * 50 + "\n")

# Delete meshes with "geometry" in their name
geometry_objects = [
    obj
    for obj in bpy.context.scene.objects
    if "geometry" in obj.name.lower() and obj.type == "MESH"
]
deleted_count = 0
for obj in geometry_objects:
    obj_name = obj.name  # Save name before deletion
    bpy.data.objects.remove(obj, do_unlink=True)
    print(f"Deleted mesh: {obj_name}")
    deleted_count += 1
if deleted_count > 0:
    print(f"Deleted {deleted_count} mesh(es) with 'geometry' in name")


# Save as insert_object4.blend
if os.path.exists(target_blend_path):
    os.remove(target_blend_path)
bpy.ops.wm.save_as_mainfile(filepath=target_blend_path)
print(f"Saved: {target_blend_path}")
