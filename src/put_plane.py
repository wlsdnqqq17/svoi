import argparse
import colorsys
import glob
import json
import math
import random
import sys
from pathlib import Path

import bpy
import bpy_extras
import cv2
from mathutils import Euler, Vector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D object insertion dataset using Blender"
    )
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID")
    parser.add_argument(
        "--glb_max_size", type=float, default=1.0, help="Maximum size for GLB objects"
    )
    parser.add_argument(
        "--primitive_max_size",
        type=float,
        default=0.6,
        help="Maximum size for primitive objects",
    )
    parser.add_argument("--plane_size", type=float, default=5.0, help="Plane size")
    parser.add_argument(
        "--safe_margin", type=float, default=0.3, help="Safe margin from plane edges"
    )
    parser.add_argument("--render_samples", type=int, default=64, help="Render samples")
    parser.add_argument(
        "--num_primitives",
        type=int,
        nargs=2,
        default=[3, 5],
        help="Range for number of primitives [min max]",
    )

    if "--" in sys.argv:
        blender_args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        blender_args = sys.argv[1:] if len(sys.argv) > 1 else []

    return parser.parse_args(blender_args)


def create_material_nodes(material_name, high_metallic=False):
    """공통 머티리얼 노드 생성 함수

    Args:
        material_name: 머티리얼 이름
        high_metallic: True면 높은 메탈릭(0.9-1.0), 낮은 러프니스(0.1-0.3) 적용
    """
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    if mat.node_tree is None:
        raise RuntimeError("Failed to create node tree")
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 기존 노드 제거
    for node in nodes:
        nodes.remove(node)

    # Principled BSDF 노드 추가
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # Output 노드 추가
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (200, 0)

    # 색상 설정 (기본값 또는 랜덤)
    hue = random.uniform(0, 1)
    saturation = random.uniform(0.3, 0.8)
    value = random.uniform(0.6, 1.0)
    base_color = (*colorsys.hsv_to_rgb(hue, saturation, value), 1.0)

    # 메탈릭/러프니스 설정
    if high_metallic:
        metallic = random.uniform(0.9, 1.0)
        roughness = random.uniform(0.1, 0.3)
    else:
        metallic = random.uniform(0, 1)
        roughness = random.uniform(0.1, 0.9)

    # 노드 값 설정
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness

    # 노드 연결
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


def create_primitive_object(primitive_type, location=(0, 0, 0), high_metallic=False):
    """Blender 내장 primitive 객체 생성 함수

    Args:
        primitive_type: primitive 타입 (cube, sphere, cylinder, cone, torus, monkey)
        location: 생성 위치
        high_metallic: True면 높은 메탈릭 머티리얼 적용
    """
    primitive_ops = {
        "cube": bpy.ops.mesh.primitive_cube_add,
        "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
        "cylinder": bpy.ops.mesh.primitive_cylinder_add,
        "cone": bpy.ops.mesh.primitive_cone_add,
        "torus": bpy.ops.mesh.primitive_torus_add,
        "monkey": bpy.ops.mesh.primitive_monkey_add,
    }

    if primitive_type not in primitive_ops:
        raise ValueError(f"Unknown primitive type: {primitive_type}")

    primitive_ops[primitive_type](location=location)
    obj = bpy.context.active_object
    if obj is None or obj.data is None:
        raise RuntimeError(f"Failed to create primitive object: {primitive_type}")

    # 머티리얼 생성 및 적용
    mat = create_material_nodes(
        f"{primitive_type}_material", high_metallic=high_metallic
    )

    # 머티리얼 적용
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    # Smooth shading 적용
    if not bpy.context.view_layer:
        raise RuntimeError
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    return obj


def export_object_as_glb(obj, export_path):
    """객체를 GLB 파일로 내보내기

    Args:
        obj: 내보낼 Blender 객체
        export_path: GLB 파일 저장 경로
    """
    view_layer = bpy.context.view_layer
    assert isinstance(view_layer, bpy.types.ViewLayer)
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")

    # Select only the target object
    obj.select_set(True)
    view_layer.objects.active = obj

    # Export as GLB
    bpy.ops.export_scene.gltf(
        filepath=str(export_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )
    print(f"Exported insertion object to: {export_path}")

    # Deselect
    obj.select_set(False)


def get_all_mesh_children(obj):
    """재귀적으로 모든 메시 자식 객체 찾기"""
    meshes = []
    if obj.type == "MESH":
        meshes.append(obj)
    for child in obj.children:
        meshes.extend(get_all_mesh_children(child))
    return meshes


def calculate_scene_bounds():
    """모든 객체의 바운딩 박스 계산"""
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name != "Plane":
            # 월드 스페이스 바운딩 박스 계산
            bbox_corners = [
                obj.matrix_world @ Vector(corner) for corner in obj.bound_box
            ]
            for corner in bbox_corners:
                min_x = min(min_x, corner.x)
                min_y = min(min_y, corner.y)
                min_z = min(min_z, corner.z)
                max_x = max(max_x, corner.x)
                max_y = max(max_y, corner.y)
                max_z = max(max_z, corner.z)

    center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
    size = Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    return center, size


def setup_camera(center, camera_name, position_offset):
    """카메라 설정"""
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    if camera is None:
        raise RuntimeError("Failed to create camera")
    camera.name = camera_name

    # Set focal length to 19mm
    if camera.data is not None and hasattr(camera.data, "lens"):
        camera.data.lens = 19
        if hasattr(camera.data, "lens_unit"):
            camera.data.lens_unit = "MILLIMETERS"

    camera_distance = 1.5
    camera.location = center + Vector(position_offset) * camera_distance

    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    if not bpy.context.scene:
        raise RuntimeError
    bpy.context.scene.camera = camera

    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    return camera


def convert_png_to_jpg(png_path, jpg_path, quality=95):
    """PNG 이미지를 JPG로 변환"""
    try:
        # PNG 이미지 읽기
        img = cv2.imread(str(png_path))
        # JPG로 저장 (품질 설정)
        success = cv2.imwrite(str(jpg_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            print(f"Converted PNG to JPG: {jpg_path}")
            return True
        else:
            print(f"Warning: Failed to save JPG file: {jpg_path}")
            return False
    except Exception as e:
        print(f"Error converting PNG to JPG: {e}")
        return False


def get_insertion_objects(insertion_object):
    """Insertion object와 모든 자식 객체들을 리스트로 반환"""
    if not insertion_object or "name" not in insertion_object:
        return []

    insertion_obj = bpy.data.objects.get(insertion_object["name"])
    if not insertion_obj:
        return []

    objects = []

    def collect_children(obj):
        objects.append(obj)
        for child in obj.children:
            collect_children(child)

    if insertion_obj.type == "EMPTY":
        collect_children(insertion_obj)
    else:
        objects.append(insertion_obj)

    return objects


def render_scene(
    camera,
    render_filename,
    dataset_dir,
    insertion_object=None,
    object_only=False,
    hide_insertion=False,
):
    if not bpy.context.scene:
        raise RuntimeError("No scene in context")

    insertion_objects = (
        get_insertion_objects(insertion_object) if insertion_object else []
    )
    hidden_from_camera = []

    # 렌더링 전 상태 설정
    if object_only:
        if not insertion_objects:
            print("Warning: No insertion object for object_only render")
            return
        for obj in bpy.data.objects:
            if obj not in insertion_objects and obj != camera:
                if obj.type in ["MESH", "EMPTY", "CURVE", "SURFACE", "META", "FONT"]:
                    obj.visible_camera = False
                    hidden_from_camera.append(obj)
        bpy.context.scene.render.film_transparent = True

    elif hide_insertion:
        if not insertion_objects:
            print("Warning: No insertion object for hide_insertion render")
            return
        for obj in insertion_objects:
            obj.hide_render = True
        bpy.context.scene.render.film_transparent = False

    else:
        bpy.context.scene.render.film_transparent = False

    bpy.context.scene.render.filepath = str(dataset_dir / render_filename)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {render_filename}")

    if object_only:
        for obj in hidden_from_camera:
            obj.visible_camera = True

    elif hide_insertion:
        for obj in insertion_objects:
            obj.hide_render = False


def find_texture(tex_dir, texture_type):
    """텍스처 파일 찾기"""
    pattern = str(tex_dir / f"{texture_type}.*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    else:
        raise FileNotFoundError(f"No texture file found for pattern: {pattern}")


def setup_world_environment(dataset_dir):
    """월드 환경 설정"""
    if not bpy.context.scene:
        raise RuntimeError
    world = bpy.context.scene.world
    if world is None:
        raise RuntimeError("No world in scene")
    world.use_nodes = True
    if world.node_tree is None:
        raise RuntimeError("Failed to create world node tree")
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links

    # 기존 노드들 제거
    for node in world_nodes:
        world_nodes.remove(node)

    # World Output 노드 추가
    world_output = world_nodes.new(type="ShaderNodeOutputWorld")
    world_output.location = (300, 0)

    # Environment Texture 노드 추가
    env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.location = (0, 0)

    # Environment Texture 이미지 로드
    envmap_pattern = str(dataset_dir / "envmap.*")
    envmap_matches = glob.glob(envmap_pattern)
    if envmap_matches:
        env_image_path = envmap_matches[0]
        env_tex.image = bpy.data.images.load(env_image_path)
        print(f"World environment map loaded: {env_image_path}")

        # Environment Texture을 World Output에 연결
        world_links.new(env_tex.outputs["Color"], world_output.inputs["Surface"])

        # 환경맵 이미지 패킹
        if env_tex.image and not env_tex.image.packed_file:
            env_tex.image.pack()
    else:
        raise FileNotFoundError(f"Environment map not found in {dataset_dir}/")


def setup_render_settings(render_samples):
    """렌더 설정"""
    if not bpy.context.scene:
        raise RuntimeError
    bpy.context.scene.render.engine = "CYCLES"
    if not bpy.context.scene.cycles:
        raise RuntimeError
    bpy.context.scene.cycles.samples = render_samples
    bpy.context.scene.cycles.preview_samples = render_samples
    print(f"Render engine set to Cycles with {render_samples} samples")


def select_camera_angle(insertion_pos: tuple[float, float] | None = None):
    """카메라 각도 선택

    Args:
        insertion_pos: insertion object의 (x, y) 위치.
                       주어지면 해당 방향의 각도를 선택.
    """
    diagonal_angles = [
        (1.8, 0.8, 0.3),
        (-1.8, 0.8, 0.3),
        (0.8, 1.8, 0.3),
        (-0.8, 1.8, 0.3),
        (1.8, -0.8, 0.3),
        (-1.8, -0.8, 0.3),
        (0.8, -1.8, 0.3),
        (-0.8, -1.8, 0.3),
        (1.5, 1.5, 0.7),
        (-1.5, 1.5, 0.7),
        (1.5, -1.5, 0.7),
        (-1.5, -1.5, 0.7),
    ]

    if insertion_pos is None:
        selected_angle = random.choice(diagonal_angles)
    else:
        # insertion object와 같은 방향(사분면)의 각도만 필터링
        ix, iy = insertion_pos
        matching_angles = [
            angle
            for angle in diagonal_angles
            if (angle[0] > 0) == (ix > 0) and (angle[1] > 0) == (iy > 0)
        ]
        selected_angle = (
            random.choice(matching_angles)
            if matching_angles
            else random.choice(diagonal_angles)
        )

    return (selected_angle[0], selected_angle[1], selected_angle[2])


def adjust_plane_material_scale(plane, scale_factor):
    """평면의 머티리얼 UV 스케일을 조정하여 텍스처 크기를 유지"""
    if plane.data.materials:
        mat = plane.data.materials[0]
        if mat and mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.name == "Mapping":
                    node.inputs["Scale"].default_value = (
                        scale_factor[0],
                        scale_factor[1],
                        1,
                    )
                    print(f"Adjusted Plane material UV scale to: {scale_factor}")
                    return
    print(
        "Warning: Could not find Mapping node in Plane material or no material found."
    )


def create_object_list(glb_paths, args):
    """GLB 파일들과 primitive 객체들을 섞어서 배치할 리스트 생성"""
    primitive_types = ["cube", "sphere", "cylinder", "cone", "torus"]
    all_objects = []

    for glb_path in glb_paths:
        all_objects.append(("glb", str(glb_path)))

    num_primitives = random.randint(args.num_primitives[0], args.num_primitives[1])
    for _ in range(num_primitives):
        primitive_type = random.choice(primitive_types)
        all_objects.append(("primitive", primitive_type))

    # 리스트 섞기
    random.shuffle(all_objects)
    return all_objects


def calculate_object_dimensions(parent_object):
    """객체의 실제 크기 계산"""
    if parent_object.type == "EMPTY":
        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")

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

        if min_x != float("inf"):
            total_width = max_x - min_x
            total_height = max_y - min_y
            total_depth = max_z - min_z
            current_dim = max(total_width, total_height, total_depth)
        else:
            current_dim = 1.0  # fallback
    else:
        current_dim = max(
            parent_object.dimensions.x,
            parent_object.dimensions.y,
            parent_object.dimensions.z,
        )

    return current_dim


def place_object_on_plane(parent_object, plane, x_pos, y_pos):
    """객체를 평면 위에 정확히 배치"""
    # 먼저 객체를 임시로 배치
    parent_object.location = (x_pos, y_pos, 0)

    # 스케일링 후 객체의 실제 바운딩 박스 계산
    if not bpy.context.view_layer:
        raise RuntimeError
    bpy.context.view_layer.update()

    plane_height = plane.location.z + (plane.dimensions.z / 2)

    if parent_object.type == "EMPTY":
        mesh_children = get_all_mesh_children(parent_object)
        if mesh_children:
            min_z_world = float("inf")
            for mesh_obj in mesh_children:
                for vertex in mesh_obj.data.vertices:
                    world_vertex = mesh_obj.matrix_world @ vertex.co
                    min_z_world = min(min_z_world, world_vertex.z)

            if min_z_world != float("inf"):
                z_offset = plane_height - min_z_world
            else:
                z_offset = plane_height
        else:
            z_offset = plane_height
    else:
        # 메시 객체의 경우 월드 스페이스 바운딩 박스 계산
        bbox_corners = [
            parent_object.matrix_world @ Vector(corner)
            for corner in parent_object.bound_box
        ]
        min_z_world = min(corner.z for corner in bbox_corners)
        z_offset = plane_height - min_z_world

    # 객체를 평면 위에 정확히 배치 (밑바닥 기준)
    parent_object.location = (x_pos, y_pos, parent_object.location.z + z_offset)


def place_single_object(
    obj_type,
    obj_data,
    index,
    placed_positions,
    target_max_dim,
    plane_size,
    safe_margin,
    plane,
    high_metallic=False,
):
    parent_object = None
    glb_file_name = None

    if obj_type == "glb":
        # GLB 파일 임포트
        existing_objects = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=obj_data)
        new_objects = set(bpy.data.objects) - existing_objects

        # 상위 객체 찾기
        for obj in new_objects:
            if obj.parent is None and (
                obj.type == "EMPTY"
                or any(child in new_objects for child in obj.children)
            ):
                parent_object = obj
                break

        if parent_object is None:
            mesh_objects = [obj for obj in new_objects if obj.type == "MESH"]
            if mesh_objects:
                parent_object = mesh_objects[0]

        if parent_object:
            glb_file_name = Path(obj_data).name

    elif obj_type == "primitive":
        parent_object = create_primitive_object(
            obj_data, location=(0, 0, 0), high_metallic=high_metallic
        )
        parent_object.name = f"{obj_data}_{index}"

    if parent_object is None:
        print(f"Object {index + 1}: No parent object found for {obj_type}: {obj_data}")
        return None, None, None

    # 객체 크기 계산 및 스케일링
    current_dim = calculate_object_dimensions(parent_object)
    scale_factor = target_max_dim / current_dim if current_dim > 0 else 1.0

    original_scale = parent_object.scale
    parent_object.scale = (
        original_scale[0] * scale_factor,
        original_scale[1] * scale_factor,
        original_scale[2] * scale_factor,
    )

    # 회전 적용
    random_rotation = random.uniform(0, 2 * math.pi)
    if obj_type == "glb":
        selected_objects = bpy.context.selected_objects
        target_obj = selected_objects[0] if selected_objects else parent_object
        target_obj.rotation_mode = "XYZ"
        original_rotation = target_obj.rotation_euler
        target_obj.rotation_euler = Euler(
            (
                original_rotation.x,
                original_rotation.y,
                original_rotation.z + random_rotation,
            ),
            "XYZ",
        )
    else:
        parent_object.rotation_euler = Euler((0, 0, random_rotation), "XYZ")

    # 스케일링 반영
    parent_object.location = (0, 0, 0)
    if not bpy.context.view_layer:
        raise RuntimeError
    bpy.context.view_layer.update()

    # 실제 스케일링된 객체의 크기 계산
    actual_dimensions = calculate_object_dimensions(parent_object)

    # 겹치지 않는 위치 찾기
    x_pos, y_pos = find_non_overlapping_position(
        actual_dimensions, placed_positions, plane_size, safe_margin
    )

    # 객체를 평면 위에 배치
    place_object_on_plane(parent_object, plane, x_pos, y_pos)

    print(
        f"Placed {obj_type} '{parent_object.name}' at ({x_pos:.2f}, {y_pos:.2f}, {parent_object.location.z:.2f})"
    )

    return parent_object, glb_file_name, (x_pos, y_pos, actual_dimensions)


def find_non_overlapping_position(
    obj_size,
    placed_positions,
    plane_size,
    safe_margin,
    max_attempts=100,
):
    best_pos = None
    best_min_distance = -1

    for _ in range(max_attempts):
        x_pos = random.uniform(
            -plane_size / 2 + safe_margin, plane_size / 2 - safe_margin
        )
        y_pos = random.uniform(
            -plane_size / 2 + safe_margin, plane_size / 2 - safe_margin
        )

        # 경계 체크
        obj_radius = obj_size * 0.6
        if not (
            abs(x_pos) + obj_radius <= plane_size / 2 - safe_margin
            and abs(y_pos) + obj_radius <= plane_size / 2 - safe_margin
        ):
            continue

        # 각 기존 객체와의 거리 계산
        min_clearance = float("inf")
        for prev_x, prev_y, prev_obj_size in placed_positions:
            distance = math.sqrt((x_pos - prev_x) ** 2 + (y_pos - prev_y) ** 2)
            required_distance = (obj_size + prev_obj_size) * 0.6
            clearance = distance - required_distance
            min_clearance = min(min_clearance, clearance)

        # 충분한 여유 공간이 있으면 바로 반환
        if min_clearance > 0:
            return x_pos, y_pos

        # 가장 덜 겹치는 위치 저장 (fallback용)
        if min_clearance > best_min_distance:
            best_min_distance = min_clearance
            best_pos = (x_pos, y_pos)

    # fallback: 가장 덜 겹치는 위치 반환
    if best_pos:
        print(
            f"Warning: Could not find non-overlapping position, using best available (clearance: {best_min_distance:.2f})"
        )
        return best_pos

    return (0, 0)


def prepare_insertion_object(insertion_object_info):
    """Insertion object 정보를 미리 준비 (렌더링 전 가시성 체크용)

    Args:
        insertion_object_info: dict with 'name' and 'file' keys
    """
    if not insertion_object_info:
        print("Warning: No insertion object info provided")
        return None

    target_name = insertion_object_info["name"]
    target_file = insertion_object_info["file"]
    parent_object = bpy.data.objects.get(target_name)

    if not parent_object:
        print(f"Warning: Object {target_name} not found in scene")
        return None

    print(
        f"Selected insertion object: {target_name}, file: {target_file}, position: ({parent_object.location.x:.2f}, {parent_object.location.y:.2f})"
    )

    # 객체 정보를 dict로 반환
    return {
        "name": target_name,
        "file": target_file,
        "location": [
            parent_object.location.x,
            parent_object.location.y,
            parent_object.location.z,
        ],
        "rotation": list(parent_object.rotation_euler),
        "scale": list(parent_object.scale),
    }


def world_to_pixel_coordinates(world_location, camera, scene):
    """3D 월드 좌표를 2D 픽셀 좌표로 변환"""
    # Blender의 world_to_camera_view 함수 사용
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, world_location)

    # 정규화된 좌표를 픽셀 좌표로 변환
    render_scale = scene.render.resolution_percentage / 100
    width = int(scene.render.resolution_x * render_scale)
    height = int(scene.render.resolution_y * render_scale)

    pixel_x = co_2d.x * width
    pixel_y = (1.0 - co_2d.y) * height  # Y축 뒤집기 (Blender는 상단이 0)
    depth = co_2d.z

    return pixel_x, pixel_y, depth


def save_metadata(args, dataset_dir, camera, insertion_object):
    """메타데이터 저장"""
    metadata = {
        "camera_location": list(camera.location),
        "camera_rotation": [math.degrees(r) for r in camera.rotation_euler],
        "insertion_object": None,
    }

    # 3D 좌표를 2D 픽셀 좌표로 변환
    world_pos = Vector(insertion_object["location"])
    pixel_x, pixel_y, depth = world_to_pixel_coordinates(
        world_pos, camera, bpy.context.scene
    )

    metadata["insertion_object"] = {
        "file": insertion_object["file"],
        "location": insertion_object["location"],
        "rotation": [
            math.degrees(r) for r in insertion_object["rotation"]
        ],  # radians를 degrees로 변환
        "scale": insertion_object["scale"],
        "pixel_coordinates": [pixel_x, pixel_y],
        "depth": depth,
    }
    print(f"Saved removed object info: {insertion_object['file']}")
    print(f"Object location: {insertion_object['location']}")
    print(f"Pixel coordinates: ({pixel_x:.1f}, {pixel_y:.1f}), depth: {depth:.3f}")

    metadata_path = dataset_dir / f"{args.dataset_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


def setup_plane_material_nodes(
    mat: bpy.types.Material,
    plane: bpy.types.Object,
    color_path: str,
    roughness_path: str,
    normal_path: str,
) -> None:
    """평면 머티리얼 노드를 설정합니다."""
    mat.use_nodes = True
    if mat.node_tree is None:
        raise RuntimeError("Failed to create node tree")
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    if plane.data is None:
        raise RuntimeError("Failed to load plane")
    for node in nodes:
        nodes.remove(node)

    # 노드 생성
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    tex_image_color = nodes.new(type="ShaderNodeTexImage")
    tex_image_rough = nodes.new(type="ShaderNodeTexImage")
    tex_image_normal = nodes.new(type="ShaderNodeTexImage")
    normal_map = nodes.new(type="ShaderNodeNormalMap")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")

    # 이미지 로드
    tex_image_color.image = bpy.data.images.load(color_path)
    tex_image_rough.image = bpy.data.images.load(roughness_path)
    if not tex_image_rough.image.colorspace_settings:
        raise RuntimeError
    tex_image_rough.image.colorspace_settings.name = "Non-Color"
    tex_image_normal.image = bpy.data.images.load(normal_path)
    if not tex_image_normal.image.colorspace_settings:
        raise RuntimeError
    tex_image_normal.image.colorspace_settings.name = "Non-Color"

    # 노드 위치 설정
    tex_coord.location = (-800, 0)
    mapping.location = (-600, 0)
    tex_image_color.location = (-400, 200)
    tex_image_rough.location = (-400, 0)
    tex_image_normal.location = (-400, -200)
    normal_map.location = (-200, -200)
    bsdf.location = (0, 0)
    output.location = (200, 0)

    # 노드 연결
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex_image_color.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex_image_rough.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex_image_normal.inputs["Vector"])
    links.new(tex_image_color.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex_image_rough.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(tex_image_normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])


def main():
    args = parse_args()

    root_dir = Path("/Users/jinwoo/Documents/work/svoi")
    dataset_dir = root_dir / "dataset" / args.dataset_id
    tex_dir = dataset_dir / "textures"
    glb_paths = [dataset_dir / f"{i}.glb" for i in range(1, 4)]

    # Randomly select insertion object type (equal probability)
    # Options: 3.glb + 5 primitives = 6 options
    insertion_options = ["3.glb", "cube", "sphere", "cylinder", "cone", "torus"]
    selected_insertion_type = random.choice(insertion_options)
    print(f"Selected insertion object type: {selected_insertion_type}")

    # Clear existing objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

    # Set texture paths
    color_path = find_texture(tex_dir, "diff")
    roughness_path = find_texture(tex_dir, "rou")
    normal_path = find_texture(tex_dir, "nor")

    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    plane = bpy.context.active_object
    if plane is None:
        raise RuntimeError
    plane.name = "Plane"

    # 평면 머티리얼 설정
    mat = bpy.data.materials.new(name="Plane_Material")
    setup_plane_material_nodes(mat, plane, color_path, roughness_path, normal_path)

    # 머티리얼 적용
    if not plane.data:
        raise RuntimeError
    if len(plane.data.materials) == 0:
        plane.data.materials.append(mat)
    else:
        plane.data.materials[0] = mat

    # 이미지 패킹
    for image in bpy.data.images:
        if not image.packed_file:
            image.pack()

    # 객체 배치
    is_primitive_insertion = selected_insertion_type != "3.glb"
    if is_primitive_insertion:
        # Exclude 3.glb when primitive is chosen as insertion object
        glb_paths_to_use = [p for p in glb_paths if not str(p).endswith("3.glb")]
    else:
        glb_paths_to_use = glb_paths

    all_objects = create_object_list(glb_paths_to_use, args)

    if is_primitive_insertion:
        # Add insertion primitive with special marker
        all_objects.append(("insertion_primitive", selected_insertion_type))

    placed_positions = []
    glb_object_to_file = {}
    insertion_object_info = None  # Track insertion object info

    insertion_pos = None
    for i, (obj_type, obj_data) in enumerate(all_objects):
        # Check if this is the insertion object
        is_insertion = False
        high_metallic = False

        if obj_type == "insertion_primitive":
            # This is the insertion primitive
            is_insertion = True
            high_metallic = True
            obj_type = "primitive"  # Treat as primitive for placement
        elif (
            obj_type == "glb"
            and obj_data.endswith("3.glb")
            and not is_primitive_insertion
        ):
            # This is 3.glb and we selected 3.glb as insertion
            is_insertion = True

        target_dim = args.glb_max_size if obj_type == "glb" else args.primitive_max_size
        parent_obj, glb_file, pos_info = place_single_object(
            obj_type,
            obj_data,
            i,
            placed_positions,
            target_dim,
            args.plane_size,
            args.safe_margin,
            plane,
            high_metallic=high_metallic,
        )
        if pos_info:
            placed_positions.append(pos_info)
        else:
            raise RuntimeError

        if parent_obj and glb_file:
            glb_object_to_file[parent_obj.name] = glb_file

        # Track insertion object info
        if is_insertion and parent_obj:
            insertion_pos = (pos_info[0], pos_info[1])
            if high_metallic:
                # Primitive insertion object - export as GLB
                glb_filename = f"{selected_insertion_type}_metallic.glb"
                export_path = dataset_dir / glb_filename
                export_object_as_glb(parent_obj, export_path)
                insertion_object_info = {
                    "name": parent_obj.name,
                    "file": glb_filename,
                }
            else:
                # GLB insertion object (3.glb)
                insertion_object_info = {
                    "name": parent_obj.name,
                    "file": glb_file,
                }

    selected_angle = select_camera_angle(insertion_pos)
    camera = setup_camera(Vector((0, 0, 0)), "Camera_Diagonal", selected_angle)

    # 환경 및 렌더 설정
    setup_world_environment(dataset_dir)
    setup_render_settings(args.render_samples)

    # Insertion object 준비
    insertion_object = prepare_insertion_object(insertion_object_info)

    # 렌더링
    render_scene(camera, f"{args.dataset_id}_after.png", dataset_dir)
    render_scene(
        camera,
        f"{args.dataset_id}_object.png",
        dataset_dir,
        insertion_object=insertion_object,
        object_only=True,
    )
    render_scene(
        camera,
        f"{args.dataset_id}_before.png",
        dataset_dir,
        insertion_object=insertion_object,
        hide_insertion=True,
    )
    convert_png_to_jpg(
        dataset_dir / f"{args.dataset_id}_before.png",
        dataset_dir / f"{args.dataset_id}_before.jpg",
    )
    print(f"Selected diagonal angle: {selected_angle}")
    print("Object insertion dataset completed!")

    # 메타데이터 저장 및 블렌더 파일 저장
    save_metadata(args, dataset_dir, camera, insertion_object)

    blend_path = dataset_dir / f"{args.dataset_id}.blend"
    if blend_path.exists():
        blend_path.unlink()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))


if __name__ == "__main__":
    main()
