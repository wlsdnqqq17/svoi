import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import (  # pyright: ignore reportMissingImports=false
    world_to_camera_view,
)
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render 3 images (before, after, object) and metadata from a blend file"
    )
    parser.add_argument("--dataset_id", type=str, required=True, help="Experiment ID")
    parser.add_argument(
        "--insertion_object_name",
        type=str,
        default="Sketchfab_model.002",
        help="Name of the insertion object",
    )
    parser.add_argument("--render_samples", type=int, default=64, help="Render samples")

    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = sys.argv[1:]

    return parser.parse_args(argv)


def render_scene(camera, filepath):
    """Render the current scene to the specified filepath"""
    bpy.context.scene.render.filepath = str(filepath)  # pyright: ignore reportMissingImports=false
    bpy.ops.render.render(write_still=True)  # pyright: ignore reportMissingImports=false
    print(f"Rendered: {filepath}")


def world_to_pixel_coordinates(world_location, camera, scene):
    """Convert 3D world coordinates to 2D pixel coordinates"""
    co_2d = world_to_camera_view(scene, camera, world_location)

    render_scale = scene.render.resolution_percentage / 100
    width = int(scene.render.resolution_x * render_scale)
    height = int(scene.render.resolution_y * render_scale)

    pixel_x = co_2d.x * width
    pixel_y = (1.0 - co_2d.y) * height
    depth = co_2d.z

    return pixel_x, pixel_y, depth


def get_all_children(obj):
    """Recursively get all children of an object"""
    children = []
    for child in obj.children:
        children.append(child)
        children.extend(get_all_children(child))
    return children


def main():
    # 1. Parse Arguments (before opening blend file to avoid context issues if possible,
    # but bpy.ops.wm.open_mainfile resets everything, so we parse first)
    # Note: When running with blender, sys.argv includes blender args.
    # The parse_args function handles the '--' separator.
    args = parse_args()
    id = args.dataset_id
    blend_file = Path(f"dataset/{id}/{id}.blend")
    output_dir = Path(f"dataset/{id}/")

    # 2. Open Blend File
    if not blend_file.exists():
        print(f"Error: Blend file not found at {blend_file}")
        sys.exit(1)

    bpy.ops.wm.open_mainfile(filepath=str(blend_file))

    # 3. Setup Render Settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.render_samples
    scene.cycles.preview_samples = args.render_samples

    # 4. Find Objects
    insertion_obj = bpy.data.objects.get(args.insertion_object_name)
    if not insertion_obj:
        print(
            f"Error: Insertion object '{args.insertion_object_name}' not found in scene"
        )
        # Try to find by partial match or list available objects?
        # For now, strict match.
        sys.exit(1)

    camera = scene.camera
    if not camera:
        print("Error: No active camera in scene")
        sys.exit(1)

    # Identify insertion object hierarchy (if it has children)
    insertion_family = [insertion_obj] + get_all_children(insertion_obj)

    # 5. Render Sequence

    # A. Render AFTER (Everything visible)
    # Ensure insertion object is visible
    for obj in insertion_family:
        obj.hide_render = False

    scene.render.film_transparent = False
    render_scene(camera, output_dir / f"{args.dataset_id}_after.png")

    # B. Render OBJECT (Only insertion object visible, transparent background)
    # Hide everything else from camera
    hidden_from_camera = []
    for obj in bpy.data.objects:
        if obj not in insertion_family and obj != camera:
            if not obj.hide_render:  # Only touch objects that are currently renderable
                # Check if object type is suitable for hiding (mesh, empty, etc)
                if obj.type in [
                    "MESH",
                    "EMPTY",
                    "CURVE",
                    "SURFACE",
                    "META",
                    "FONT",
                    "HAIR",
                    "POINTCLOUD",
                    "VOLUME",
                    "GPENCIL",
                ]:
                    obj.visible_camera = False
                    hidden_from_camera.append(obj)

    scene.render.film_transparent = True
    render_scene(camera, output_dir / f"{args.dataset_id}_object.png")

    # Restore visibility
    for obj in hidden_from_camera:
        obj.visible_camera = True

    # C. Render BEFORE (Insertion object hidden)
    for obj in insertion_family:
        obj.hide_render = True

    scene.render.film_transparent = False
    render_scene(camera, output_dir / f"{args.dataset_id}_before.png")
    img = Image.open(output_dir / f"{args.dataset_id}_before.png").convert("RGB")
    img.save(output_dir / f"{args.dataset_id}_before.jpg", quality=95)

    # Restore insertion object visibility (good practice)
    for obj in insertion_family:
        obj.hide_render = False

    # 6. Generate Metadata
    pixel_x, pixel_y, depth = world_to_pixel_coordinates(
        insertion_obj.location, camera, scene
    )

    metadata = {
        "dataset_id": args.dataset_id,
        "camera_location": list(camera.location),
        "camera_rotation": [math.degrees(x) for x in camera.rotation_euler],
        "insertion_object": {
            "file": "3.glb",
            "name": insertion_obj.name,
            "location": list(insertion_obj.location),
            "rotation": [math.degrees(x) for x in insertion_obj.rotation_euler],
            "scale": list(insertion_obj.scale),
            "pixel_coordinates": [pixel_x, pixel_y],
            "depth": depth,
        },
    }

    metadata_path = output_dir / f"{args.dataset_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")
    print("Done!")


if __name__ == "__main__":
    main()
