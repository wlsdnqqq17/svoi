import bpy
import random
import sys
import math
import glob
import argparse
import json
from mathutils import Vector, Euler
from pathlib import Path
import colorsys
from bpy_extras.object_utils import world_to_camera_view

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D object insertion dataset using Blender")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID")
    parser.add_argument("--glb_max_size", type=float, default=1.0, help="Maximum size for GLB objects (default: 1.0)")
    parser.add_argument("--primitive_max_size", type=float, default=0.6, help="Maximum size for primitive objects (default: 0.6)")
    parser.add_argument("--plane_size", type=float, default=5.0, help="Plane size (default: 5.0)")
    parser.add_argument("--safe_margin", type=float, default=0.3, help="Safe margin from plane edges (default: 0.3)")
    parser.add_argument("--render_samples", type=int, default=64, help="Render samples (default: 64)")
    parser.add_argument("--num_primitives", type=int, nargs=2, default=[3, 5], help="Range for number of primitives [min max] (default: 3 5)")
    parser.add_argument("--use_chrome_ball", action="store_true", help="Use chrome ball as insertion object instead of GLB object (default: False)")
    
    # Blender에서 실행될 때는 sys.argv에서 Blender 관련 인자들을 제거
    if '--' in sys.argv:
        blender_args = sys.argv[sys.argv.index('--') + 1:]
    else:
        # Blender 외부에서 실행될 때를 위한 fallback
        blender_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    return parser.parse_args(blender_args)



def create_material_nodes(material_name, base_color=None, metallic=None, roughness=None):
    """공통 머티리얼 노드 생성 함수"""
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # 기존 노드 제거
    for node in nodes:
        nodes.remove(node)
    
    # Principled BSDF 노드 추가
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Output 노드 추가
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    
    # 색상 설정 (기본값 또는 랜덤)
    if base_color is None:
        hue = random.uniform(0, 1)
        saturation = random.uniform(0.3, 0.8)
        value = random.uniform(0.6, 1.0)
        base_color = (*colorsys.hsv_to_rgb(hue, saturation, value), 1.0)
    
    # 메탈릭/러프니스 설정 (기본값 또는 랜덤)
    if metallic is None:
        metallic = random.uniform(0, 1)
    if roughness is None:
        roughness = random.uniform(0.1, 0.9)
    
    # 노드 값 설정
    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    
    # 노드 연결
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_primitive_object(primitive_type, location=(0, 0, 0)):
    """Blender 내장 primitive 객체 생성 함수"""
    primitive_ops = {
        'cube': bpy.ops.mesh.primitive_cube_add,
        'sphere': bpy.ops.mesh.primitive_uv_sphere_add,
        'cylinder': bpy.ops.mesh.primitive_cylinder_add,
        'cone': bpy.ops.mesh.primitive_cone_add,
        'torus': bpy.ops.mesh.primitive_torus_add,
        'monkey': bpy.ops.mesh.primitive_monkey_add
    }
    
    if primitive_type not in primitive_ops:
        raise ValueError(f"Unknown primitive type: {primitive_type}")
    
    primitive_ops[primitive_type](location=location)
    obj = bpy.context.active_object
    
    # 랜덤 머티리얼 생성 및 적용
    mat = create_material_nodes(f"{primitive_type}_material")
    
    # 머티리얼 적용
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    
    # Smooth shading 적용
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    
    return obj

def get_all_mesh_children(obj):
    """재귀적으로 모든 메시 자식 객체 찾기"""
    meshes = []
    if obj.type == 'MESH':
        meshes.append(obj)
    for child in obj.children:
        meshes.extend(get_all_mesh_children(child))
    return meshes

def calculate_scene_bounds():
    """모든 객체의 바운딩 박스 계산"""
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name != 'Plane':
            # 월드 스페이스 바운딩 박스 계산
            bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
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

def setup_camera(center, size, camera_name, position_offset):
    """카메라 추가 및 설정 (렌더링 없이)"""
    # 카메라 추가
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = camera_name
    
    # 씬의 크기에 따라 카메라 거리 계산 (더욱 가깝게)
    max_dim = max(size.x, size.y, size.z)
    camera_distance = max_dim * 0.9  # 더욱 가까운 거리로 조정
    
    # 카메라 위치 설정
    camera.location = center + Vector(position_offset) * camera_distance
    
    # 카메라가 중심점을 바라보도록 설정
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # 카메라를 씬의 활성 카메라로 설정
    bpy.context.scene.camera = camera
    
    # 렌더 설정 (1920x1080)
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    
    return camera

def render_scene(camera, render_filename, dataset_dir):
    """기존 카메라로 씬 렌더링"""
    bpy.context.scene.render.filepath = str(dataset_dir / render_filename)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {render_filename} from {camera.name}")



def add_chrome_ball_insertion_object(placed_positions, plane_size, safe_margin):
    """Chrome ball insertion object 추가"""
    # 구체 생성
    bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
    sphere = bpy.context.active_object
    sphere.name = "insertion_chrome_ball"
    
    # 크기 설정
    sphere_scale = 0.3
    sphere.scale = (sphere_scale, sphere_scale, sphere_scale)
    
    # 크롬 머티리얼 생성 및 적용
    chrome_color = (0.8, 0.8, 0.9, 1.0)  # 살짝 파란빛 실버
    mat = create_material_nodes("Metallic_Material", chrome_color, metallic=1.0, roughness=0.1)
    
    if len(sphere.data.materials) == 0:
        sphere.data.materials.append(mat)
    else:
        sphere.data.materials[0] = mat
    
    # Smooth shading 적용
    bpy.context.view_layer.objects.active = sphere
    bpy.ops.object.shade_smooth()
    
    # 유효한 위치 찾기 (인라인)
    for attempt in range(50):
        x_pos = random.uniform(-plane_size/2 + safe_margin, plane_size/2 - safe_margin)
        y_pos = random.uniform(-plane_size/2 + safe_margin, plane_size/2 - safe_margin)
        
        min_distance = sphere_scale * 2.0
        too_close = any(
            math.sqrt((x_pos - prev_x)**2 + (y_pos - prev_y)**2) < min_distance
            for prev_x, prev_y in placed_positions
        )
        
        if not too_close and (abs(x_pos) + sphere_scale <= plane_size/2 - safe_margin and 
                             abs(y_pos) + sphere_scale <= plane_size/2 - safe_margin):
            break
    else:
        # 적절한 위치를 못 찾으면 더 안전한 위치 사용
        x_pos = random.uniform(-plane_size/3, plane_size/3)
        y_pos = random.uniform(-plane_size/3, plane_size/3)
    
    # 평면 높이 계산 및 구체 배치
    plane = bpy.data.objects['Plane']
    plane_height = plane.location.z + (plane.dimensions.z / 2)
    sphere.location = (x_pos, y_pos, plane_height + sphere_scale)
    
    print(f"Chrome ball insertion object added at position: ({x_pos:.2f}, {y_pos:.2f}, {sphere.location.z:.2f})")
    return sphere

def hide_glb_object_for_insertion(glb_object_to_file):
    """이미 씬에 있는 GLB 객체 중 하나를 렌더링에서 숨겨서 insertion object로 사용"""
    # 현재 씬에 있는 GLB 객체들 찾기 (Sketchfab_model로 시작하는 객체들)
    glb_objects = []
    for obj in bpy.data.objects:
        if obj.name.startswith('Sketchfab_model') and obj.type in ['EMPTY', 'MESH']:
            glb_objects.append(obj)
    
    if not glb_objects:
        print("Warning: No GLB objects found in scene")
        return None
    
    # 랜덤으로 GLB 객체 선택
    target_object = random.choice(glb_objects)
    target_name = target_object.name
    target_file = glb_object_to_file.get(target_name, "unknown.glb")
    
    # 객체의 변환 행렬 정보 저장
    target_matrix_world = target_object.matrix_world.copy()
    target_location = list(target_object.location)
    target_rotation = list(target_object.rotation_euler)
    target_scale = list(target_object.scale)
    
    print(f"Selected GLB object for hiding: {target_name}, file: {target_file}, position: ({target_object.location.x:.2f}, {target_object.location.y:.2f})")
    
    # 선택된 객체와 모든 자식 객체들을 찾아서 렌더링에서 숨기기
    objects_to_hide = []
    
    if target_object.type == 'EMPTY':
        # Empty 객체의 경우 모든 자식 객체들도 수집
        def collect_children(obj):
            objects_to_hide.append(obj)
            for child in obj.children:
                collect_children(child)
        
        collect_children(target_object)
    else:
        # 메시 객체의 경우
        objects_to_hide.append(target_object)
    
    # 객체들을 렌더링에서 숨기기
    for obj in objects_to_hide:
        print(f"Hiding object from render: {obj.name}")
        obj.hide_render = True
    
    print(f"GLB object hidden successfully. Total {len(objects_to_hide)} objects hidden.")
    return {
        "name": target_name, 
        "file": target_file,
        "matrix_world": [list(row) for row in target_matrix_world],
        "location": target_location,
        "rotation": target_rotation,
        "scale": target_scale,
        "hidden_objects": objects_to_hide  # 숨긴 객체들의 참조 저장
    }



def clear_scene():
    """씬 초기화"""
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

def create_plane_with_textures(tex_dir, dataset_id, plane_size):
    """텍스처가 적용된 평면 생성"""
    # 텍스처 파일 찾기 (인라인)
    def find_texture(texture_type):
        pattern = str(tex_dir / f"{texture_type}.*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        else:
            raise FileNotFoundError(f"No texture file found for pattern: {pattern}")
    
    color_path = find_texture("diff")
    roughness_path = find_texture("rou") 
    normal_path = find_texture("nor")

    # 평면 생성
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Plane"

    # 평면 머티리얼 설정
    mat = bpy.data.materials.new(name="Plane_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    # 노드 생성
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    mapping = nodes.new(type='ShaderNodeMapping')
    tex_image_color = nodes.new(type='ShaderNodeTexImage')
    tex_image_rough = nodes.new(type='ShaderNodeTexImage')
    tex_image_normal = nodes.new(type='ShaderNodeTexImage')
    normal_map = nodes.new(type='ShaderNodeNormalMap')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    output = nodes.new(type='ShaderNodeOutputMaterial')

    # 이미지 로드
    tex_image_color.image = bpy.data.images.load(color_path)
    tex_image_rough.image = bpy.data.images.load(roughness_path)
    tex_image_rough.image.colorspace_settings.name = 'Non-Color'
    tex_image_normal.image = bpy.data.images.load(normal_path)
    tex_image_normal.image.colorspace_settings.name = 'Non-Color'

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
    links.new(tex_image_color.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(tex_image_rough.outputs['Color'], bsdf.inputs['Roughness'])
    links.new(tex_image_normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # 머티리얼 적용
    if len(plane.data.materials) == 0:
        plane.data.materials.append(mat)
    else:
        plane.data.materials[0] = mat

    # 이미지 패킹
    for image in bpy.data.images:
        if not image.packed_file:
            image.pack()
    
    return plane

def create_object_list(glb_paths, args):
    """GLB 파일들과 primitive 객체들을 섞어서 배치할 리스트 생성"""
    primitive_types = ['cube', 'sphere', 'cylinder', 'cone', 'torus', 'monkey']
    all_objects = []
    
    # 모든 GLB 파일 추가
    for glb_path in glb_paths:
        all_objects.append(('glb', str(glb_path)))

    # 파라미터로 받은 범위에서 랜덤 primitive 객체 추가
    num_primitives = random.randint(args.num_primitives[0], args.num_primitives[1])
    for _ in range(num_primitives):
        primitive_type = random.choice(primitive_types)
        all_objects.append(('primitive', primitive_type))

    # 리스트 섞기
    random.shuffle(all_objects)
    return all_objects

def calculate_object_dimensions(parent_object):
    """객체의 실제 크기 계산"""
    if parent_object.type == 'EMPTY':
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
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
    
    return current_dim

def place_object_on_plane(parent_object, plane, x_pos, y_pos):
    """객체를 평면 위에 정확히 배치"""
    # 먼저 객체를 임시로 배치
    parent_object.location = (x_pos, y_pos, 0)
    
    # 스케일링 후 객체의 실제 바운딩 박스 계산
    bpy.context.view_layer.update()  # 변형 업데이트
    
    plane_height = plane.location.z + (plane.dimensions.z / 2)
    
    if parent_object.type == 'EMPTY':
        # Empty 객체의 경우 자식들의 월드 스페이스 바운딩 박스 계산
        mesh_children = get_all_mesh_children(parent_object)
        if mesh_children:
            min_z_world = float('inf')
            for mesh_obj in mesh_children:
                for vertex in mesh_obj.data.vertices:
                    world_vertex = mesh_obj.matrix_world @ vertex.co
                    min_z_world = min(min_z_world, world_vertex.z)
            
            if min_z_world != float('inf'):
                z_offset = plane_height - min_z_world
            else:
                z_offset = plane_height
        else:
            z_offset = plane_height
    else:
        # 메시 객체의 경우 월드 스페이스 바운딩 박스 계산
        bbox_corners = [parent_object.matrix_world @ Vector(corner) for corner in parent_object.bound_box]
        min_z_world = min(corner.z for corner in bbox_corners)
        z_offset = plane_height - min_z_world
    
    # 객체를 평면 위에 정확히 배치 (밑바닥 기준)
    parent_object.location = (x_pos, y_pos, parent_object.location.z + z_offset)

def place_all_objects(glb_paths, args, target_max_dim_glb, target_max_dim_primitive, plane_size, safe_margin, plane):
    """모든 객체를 배치하는 함수"""
    all_objects = create_object_list(glb_paths, args)
    placed_positions = []
    glb_object_to_file = {}

    # 객체 배치 루프
    for i, (obj_type, obj_data) in enumerate(all_objects):
        parent_object = None
        
        if obj_type == 'glb':
            # GLB 파일 임포트
            existing_objects = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=obj_data)
            new_objects = set(bpy.data.objects) - existing_objects
            
            # 상위 객체 찾기
            for obj in new_objects:
                if obj.parent is None and (obj.type == 'EMPTY' or any(child in new_objects for child in obj.children)):
                    parent_object = obj
                    break
            
            if parent_object is None:
                mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
                if mesh_objects:
                    parent_object = mesh_objects[0]
            
            # GLB 객체와 파일 경로 매핑 저장
            if parent_object:
                glb_object_to_file[parent_object.name] = Path(obj_data).name
        
        elif obj_type == 'primitive':
            # Primitive 객체 생성
            parent_object = create_primitive_object(obj_data, location=(0, 0, 0))
            parent_object.name = f"{obj_data}_{i}"
        
        if parent_object:
            # 객체 크기 계산 및 스케일링
            current_dim = calculate_object_dimensions(parent_object)
            target_dim = target_max_dim_glb if obj_type == 'glb' else target_max_dim_primitive
            scale_factor = target_dim / current_dim if current_dim > 0 else 1.0
            parent_object.scale = (scale_factor, scale_factor, scale_factor)
            
            # insert_object.py 방식으로 회전 적용
            random_rotation = random.uniform(0, 2 * math.pi)
            if obj_type == 'glb':
                # GLB의 경우 선택된 첫 번째 객체에만 회전 적용
                selected_objects = bpy.context.selected_objects
                if selected_objects:
                    imported_obj = selected_objects[0]
                    imported_obj.rotation_mode = 'XYZ'
                    
                    # 기존 회전값 확인 및 출력
                    original_rotation = imported_obj.rotation_euler
                    print(f"GLB '{imported_obj.name}' 원본 회전값: X={math.degrees(original_rotation.x):.1f}°, Y={math.degrees(original_rotation.y):.1f}°, Z={math.degrees(original_rotation.z):.1f}°")
                    
                    # 기존 회전값에 랜덤 Z축 회전 추가
                    new_rotation = Euler((original_rotation.x, original_rotation.y, original_rotation.z + random_rotation), 'XYZ')
                    imported_obj.rotation_euler = new_rotation
                    
                    print(f"GLB '{imported_obj.name}' 새로운 회전값: X={math.degrees(new_rotation.x):.1f}°, Y={math.degrees(new_rotation.y):.1f}°, Z={math.degrees(new_rotation.z):.1f}°")
                else:
                    # fallback: 부모 객체에 회전 적용
                    original_rotation = parent_object.rotation_euler
                    print(f"Parent '{parent_object.name}' 원본 회전값: X={math.degrees(original_rotation.x):.1f}°, Y={math.degrees(original_rotation.y):.1f}°, Z={math.degrees(original_rotation.z):.1f}°")
                    new_rotation = Euler((original_rotation.x, original_rotation.y, original_rotation.z + random_rotation), 'XYZ')
                    parent_object.rotation_euler = new_rotation
            else:
                # Primitive 객체의 경우 기존 방식 유지
                parent_object.rotation_euler = Euler((0, 0, random_rotation), 'XYZ')
            
            # 위치 찾기
            current_target = target_max_dim_glb if obj_type == 'glb' else target_max_dim_primitive
            actual_scale = scale_factor * current_target
            
            # 겹치지 않는 위치 찾기 (기존 로직 재사용)
            max_attempts = 30
            for attempt in range(max_attempts):
                x_pos = random.uniform(-plane_size/2 + safe_margin, plane_size/2 - safe_margin)
                y_pos = random.uniform(-plane_size/2 + safe_margin, plane_size/2 - safe_margin)
                
                min_distance = actual_scale * 1.5
                too_close = any(
                    math.sqrt((x_pos - prev_x)**2 + (y_pos - prev_y)**2) < min_distance
                    for prev_x, prev_y in placed_positions
                )
                
                obj_radius = actual_scale * 0.6
                if (not too_close and 
                    abs(x_pos) + obj_radius <= plane_size/2 - safe_margin and 
                    abs(y_pos) + obj_radius <= plane_size/2 - safe_margin):
                    break
            
            # 적절한 위치를 못 찾으면 더 안전한 위치 사용
            if attempt == max_attempts - 1:
                x_pos = random.uniform(-plane_size/3, plane_size/3)
                y_pos = random.uniform(-plane_size/3, plane_size/3)
            
            placed_positions.append((x_pos, y_pos))
            
            # 객체를 평면 위에 배치
            place_object_on_plane(parent_object, plane, x_pos, y_pos)
            
            print(f"Object {i+1}: type='{obj_type}', name='{parent_object.name}', scale: {scale_factor:.3f}, position: ({x_pos:.2f}, {y_pos:.2f}, {parent_object.location.z:.2f}), rotation: {math.degrees(random_rotation):.1f}°")
        else:
            print(f"Object {i+1}: No parent object found for {obj_type}: {obj_data}")
    
    return placed_positions, glb_object_to_file





def render_scenes(args, dataset_dir, placed_positions, glb_object_to_file, plane_size, safe_margin):
    """씬 렌더링 (before/after)"""
    # 씬 바운드 계산
    center, size = calculate_scene_bounds()
    print(f"Scene center: {center}, Scene size: {size}")

    # 다양한 대각선 각도 정의
    diagonal_angles = [
        (1.5, 1.5, 1.2), (-1.5, 1.5, 1.2), (1.5, -1.5, 1.2), (-1.5, -1.5, 1.2),
        (2.0, 1.0, 1.5), (-2.0, 1.0, 1.5), (1.0, 2.0, 1.5), (-1.0, -2.0, 1.5),
        (1.8, 0.8, 0.8), (-1.8, 0.8, 0.8), (0.8, 1.8, 0.8), (-0.8, -1.8, 0.8),
    ]

    # 랜덤으로 대각선 각도 선택
    selected_angle = random.choice(diagonal_angles)

    # 카메라 한 번만 생성
    camera = setup_camera(center, size, "Camera_Diagonal", selected_angle)

    # 1. 먼저 insertion object 준비 (chrome ball 추가 또는 GLB 객체 제거)
    insertion_object = None
    
    if args.use_chrome_ball:
        print("Selected insertion type: Chrome Ball")
        insertion_object = add_chrome_ball_insertion_object(placed_positions, plane_size, safe_margin)
        
        # AFTER: chrome ball이 있는 상태 렌더링 (원본)
        render_scene(camera, f"{args.dataset_id}_after.png", dataset_dir)
        print(f"Rendered scene with Chrome Ball (AFTER): {args.dataset_id}_after.png")
        
        # OBJECT ONLY: chrome ball만 보이도록 다른 모든 객체 숨기기
        hidden_objects = []
        for obj in bpy.data.objects:
            if obj != insertion_object and obj != camera and obj.type in ['MESH', 'EMPTY']:
                if not obj.hide_render:
                    obj.hide_render = True
                    hidden_objects.append(obj)
                    print(f"Hiding object: {obj.name}")
        
        # 배경을 투명하게 설정
        bpy.context.scene.render.film_transparent = True
        print("Set film_transparent = True for chrome ball render")
        
        render_scene(camera, f"{args.dataset_id}_object.png", dataset_dir)
        print(f"Rendered Chrome Ball only (OBJECT): {args.dataset_id}_object.png")
        
        # 배경 투명도 원상복구
        bpy.context.scene.render.film_transparent = False
        
        # 숨겼던 객체들 다시 보이게 하기
        for obj in hidden_objects:
            obj.hide_render = False
        
        # chrome ball 렌더링에서 숨기기
        insertion_object.hide_render = True
        
        # BEFORE: chrome ball이 숨겨진 상태 렌더링 (편집됨)
        render_scene(camera, f"{args.dataset_id}_before.png", dataset_dir)
        print(f"Rendered scene without Chrome Ball (BEFORE): {args.dataset_id}_before.png")
        
    else:
        print("Selected insertion type: GLB Object (default)")
        
        # AFTER: 모든 GLB 객체가 있는 원본 상태 렌더링 (원본)
        render_scene(camera, f"{args.dataset_id}_after.png", dataset_dir)
        print(f"Rendered original scene with all objects (AFTER): {args.dataset_id}_after.png")
        
        # GLB 객체 숨기기
        hidden_object_info = hide_glb_object_for_insertion(glb_object_to_file)
        insertion_object = hidden_object_info
        
        if insertion_object:
            # OBJECT ONLY: insertion object만 보이도록 다른 모든 객체 숨기기
            # 먼저 숨겨진 insertion object를 다시 보이게 함
            insertion_object_name = insertion_object['name']
            insertion_obj = bpy.data.objects.get(insertion_object_name)
            
            if insertion_obj:
                print(f"Found insertion object: {insertion_object_name}, type: {insertion_obj.type}")
                
                # insertion object와 관련된 모든 객체들 찾기
                insertion_objects = []
                if insertion_obj.type == 'EMPTY':
                    def collect_all_children(obj):
                        insertion_objects.append(obj)
                        for child in obj.children:
                            collect_all_children(child)
                    collect_all_children(insertion_obj)
                else:
                    insertion_objects.append(insertion_obj)
                
                # 모든 insertion object들 보이게 하기
                for obj in insertion_objects:
                    obj.hide_render = False
                    print(f"Showing insertion object: {obj.name}")
                
                # 다른 모든 객체 숨기기 (insertion object 제외)
                hidden_objects = []
                for obj in bpy.data.objects:
                    if (obj not in insertion_objects and 
                        obj != camera and 
                        obj.type in ['MESH', 'EMPTY'] and 
                        not obj.hide_render):
                        obj.hide_render = True
                        hidden_objects.append(obj)
                        print(f"Hiding object: {obj.name}")
                
                # 배경을 투명하게 설정
                bpy.context.scene.render.film_transparent = True
                print("Set film_transparent = True for object-only render")
                
                render_scene(camera, f"{args.dataset_id}_object.png", dataset_dir)
                print(f"Rendered GLB insertion object only (OBJECT): {args.dataset_id}_object.png")
                
                # 배경 투명도 원상복구
                bpy.context.scene.render.film_transparent = False
                
                # 숨겼던 객체들 다시 보이게 하기
                for obj in hidden_objects:
                    obj.hide_render = False
                
                # insertion object들 다시 숨기기
                for obj in insertion_objects:
                    obj.hide_render = True
            else:
                print(f"Warning: Insertion object '{insertion_object_name}' not found in scene")
        
        # BEFORE: GLB 객체가 숨겨진 상태 렌더링 (편집됨)
        if insertion_object:
            render_scene(camera, f"{args.dataset_id}_before.png", dataset_dir)
            print(f"Rendered scene with GLB Object hidden (BEFORE) - hidden: {hidden_object_info['name']}: {args.dataset_id}_before.png")
        else:
            print("Warning: Failed to hide GLB object")

    print(f"Selected diagonal angle: {selected_angle}")
    print("Object insertion dataset completed!")
    
    return camera, insertion_object, selected_angle, center

def world_to_pixel_coordinates(world_location, camera, scene):
    """3D 월드 좌표를 2D 픽셀 좌표로 변환"""
    # Blender의 world_to_camera_view 함수 사용
    co_2d = world_to_camera_view(scene, camera, world_location)
    
    # 정규화된 좌표를 픽셀 좌표로 변환
    render_scale = scene.render.resolution_percentage / 100
    width = int(scene.render.resolution_x * render_scale)
    height = int(scene.render.resolution_y * render_scale)
    
    pixel_x = co_2d.x * width
    pixel_y = (1.0 - co_2d.y) * height  # Y축 뒤집기 (Blender는 상단이 0)
    depth = co_2d.z
    
    return pixel_x, pixel_y, depth

def save_metadata(args, dataset_dir, camera, insertion_object, selected_angle, center):
    """메타데이터 저장"""
    metadata = {
        "camera_location": list(camera.location),
        "camera_rotation": [math.degrees(r) for r in camera.rotation_euler],
        "insertion_object": None
    }
    
    # 제거된 객체 정보 저장
    if not args.use_chrome_ball and insertion_object:
        # 3D 좌표를 2D 픽셀 좌표로 변환
        world_pos = Vector(insertion_object["location"])
        pixel_x, pixel_y, depth = world_to_pixel_coordinates(world_pos, camera, bpy.context.scene)
        
        metadata["insertion_object"] = {
            "file": insertion_object["file"],
            "location": insertion_object["location"],
            "rotation": [math.degrees(r) for r in insertion_object["rotation"]],  # radians를 degrees로 변환
            "scale": insertion_object["scale"],
            "pixel_coordinates": [pixel_x, pixel_y],
            "depth": depth
        }
        print(f"Saved removed object info: {insertion_object['file']}")
        print(f"Object location: {insertion_object['location']}")
        print(f"Pixel coordinates: ({pixel_x:.1f}, {pixel_y:.1f}), depth: {depth:.3f}")
    elif args.use_chrome_ball and insertion_object:
        # 3D 좌표를 2D 픽셀 좌표로 변환
        world_pos = insertion_object.location
        pixel_x, pixel_y, depth = world_to_pixel_coordinates(world_pos, camera, bpy.context.scene)
        
        metadata["insertion_object"] = {
            "file": "chrome_ball",
            "location": list(insertion_object.location),
            "rotation": [0, 0, 0],  # chrome ball은 회전 없음
            "scale": list(insertion_object.scale),
            "pixel_coordinates": [pixel_x, pixel_y],
            "depth": depth
        }
        print("Saved chrome ball info")
        print(f"Chrome ball location: {list(insertion_object.location)}")
        print(f"Pixel coordinates: ({pixel_x:.1f}, {pixel_y:.1f}), depth: {depth:.3f}")
    
    metadata_path = dataset_dir / f"{args.dataset_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

def main():
    args = parse_args()
    
    # 디렉토리 및 파일 경로 설정 (인라인)
    root_dir = Path("/Users/jinwoo/Documents/work/svoi")
    dataset_dir = root_dir / "dataset" / args.dataset_id
    tex_dir = dataset_dir / "textures"
    
    # 디렉토리 존재 확인
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not tex_dir.exists():
        raise FileNotFoundError(f"Texture directory not found: {tex_dir}")
    
    # GLB 파일 경로 확인
    glb_paths = [dataset_dir / f"{i}.glb" for i in range(1, 4)]
    for glb_path in glb_paths:
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")
    
    # 씬 초기화 및 평면 생성
    clear_scene()
    plane = create_plane_with_textures(tex_dir, args.dataset_id, args.plane_size)

    # 객체 리스트 생성 및 배치
    placed_positions, glb_object_to_file = place_all_objects(
        glb_paths, args, args.glb_max_size, args.primitive_max_size, args.plane_size, args.safe_margin, plane
    )

    # 환경 설정 (인라인)
    world = bpy.context.scene.world
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links

    # 기존 노드들 제거
    for node in world_nodes:
        world_nodes.remove(node)

    # World Output 노드 추가
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    world_output.location = (300, 0)

    # Environment Texture 노드 추가
    env_tex = world_nodes.new(type='ShaderNodeTexEnvironment')
    env_tex.location = (0, 0)

    # Environment Texture 이미지 로드
    envmap_pattern = str(dataset_dir / "envmap.*")
    envmap_matches = glob.glob(envmap_pattern)
    if envmap_matches:
        env_image_path = envmap_matches[0]
        env_tex.image = bpy.data.images.load(env_image_path)
        print(f"World environment map loaded: {env_image_path}")
        
        # Environment Texture을 World Output에 연결
        world_links.new(env_tex.outputs['Color'], world_output.inputs['Surface'])
        
        # 환경맵 이미지 패킹
        if env_tex.image and not env_tex.image.packed_file:
            env_tex.image.pack()
            print(f"Environment map packed: {env_image_path}")
    else:
        raise FileNotFoundError(f"Environment map not found in {dataset_dir}/")

    # 렌더 설정 (인라인)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = args.render_samples
    bpy.context.scene.cycles.preview_samples = args.render_samples
    print(f"Render engine set to Cycles with {args.render_samples} samples")

    # 씬 렌더링
    camera, insertion_object, selected_angle, center = render_scenes(
        args, dataset_dir, placed_positions, glb_object_to_file, args.plane_size, args.safe_margin
    )

    # 메타데이터 저장
    save_metadata(args, dataset_dir, camera, insertion_object, selected_angle, center)

    # 블렌더 파일 저장
    blend_path = dataset_dir / f"{args.dataset_id}.blend"
    if blend_path.exists():
        blend_path.unlink()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

if __name__ == "__main__":
    main()
