import bpy
import os
import random
import sys
import math
import glob
import argparse
from mathutils import Vector, Euler
from pathlib import Path
import colorsys

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

def find_texture_file(tex_dir, dataset_id, texture_type):
    """동적으로 텍스처 파일 찾기"""
    pattern = str(tex_dir / f"{dataset_id}-{texture_type}.*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]  # 첫 번째 매치된 파일 반환
    else:
        raise FileNotFoundError(f"No texture file found for pattern: {pattern}")

def create_primitive_object(primitive_type, location=(0, 0, 0)):
    """Blender 내장 primitive 객체 생성 함수"""
    if primitive_type == 'cube':
        bpy.ops.mesh.primitive_cube_add(location=location)
    elif primitive_type == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(location=location)
    elif primitive_type == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(location=location)
    elif primitive_type == 'cone':
        bpy.ops.mesh.primitive_cone_add(location=location)
    elif primitive_type == 'torus':
        bpy.ops.mesh.primitive_torus_add(location=location)
    elif primitive_type == 'monkey':
        bpy.ops.mesh.primitive_monkey_add(location=location)
    
    obj = bpy.context.active_object
    
    # 랜덤 머티리얼 생성 및 적용
    mat = bpy.data.materials.new(name=f"{primitive_type}_material")
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
    
    # 랜덤 색상 (HSV에서 밝은 색상들)
    hue = random.uniform(0, 1)
    saturation = random.uniform(0.3, 0.8)
    value = random.uniform(0.6, 1.0)
    
    # HSV를 RGB로 변환
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # 랜덤 메탈릭 및 러프니스
    metallic = random.uniform(0, 1)
    roughness = random.uniform(0.1, 0.9)
    
    # 노드 값 설정
    bsdf.inputs['Base Color'].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    
    # 노드 연결
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # 머티리얼 적용
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    
    # Smooth shading 적용 (부드러운 표면)
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

def setup_camera_and_render(center, size, camera_name, position_offset, render_filename, dataset_dir):
    """카메라 추가 및 설정"""
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
    bpy.context.scene.render.filepath = str(dataset_dir / render_filename)
    
    # 렌더 실행
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {render_filename} from {camera_name}")
    
    return camera

def add_chrome_ball_insertion_object(placed_positions, PLANE_SIZE, SAFE_MARGIN):
    """Chrome ball insertion object 추가"""
    # 구체 생성
    bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
    sphere = bpy.context.active_object
    sphere.name = "insertion_chrome_ball"
    
    # 크기 설정 (적당한 크기)
    sphere_scale = 0.3
    sphere.scale = (sphere_scale, sphere_scale, sphere_scale)
    
    # 메탈릭 머티리얼 생성
    mat = bpy.data.materials.new(name="Metallic_Material")
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
    
    # 메탈릭 설정 (완전 메탈릭, 크롬 같은 색상)
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.9, 1.0)  # 살짝 파란빛 실버
    bsdf.inputs['Metallic'].default_value = 1.0  # 완전 메탈릭
    bsdf.inputs['Roughness'].default_value = 0.1  # 거의 거울 같음
    
    # 노드 연결
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # 머티리얼 적용
    if len(sphere.data.materials) == 0:
        sphere.data.materials.append(mat)
    else:
        sphere.data.materials[0] = mat
    
    # Smooth shading 적용
    bpy.context.view_layer.objects.active = sphere
    bpy.ops.object.shade_smooth()
    
    # 겹치지 않는 랜덤 위치 찾기
    max_attempts = 50
    for attempt in range(max_attempts):
        # 평면 범위 내에서 랜덤 위치 생성
        x_pos = random.uniform(-PLANE_SIZE/2 + SAFE_MARGIN, PLANE_SIZE/2 - SAFE_MARGIN)
        y_pos = random.uniform(-PLANE_SIZE/2 + SAFE_MARGIN, PLANE_SIZE/2 - SAFE_MARGIN)
        
        # 기존 객체들과 거리 확인
        too_close = False
        min_distance = sphere_scale * 2.0  # 구체 지름만큼 최소 거리
        
        for prev_x, prev_y in placed_positions:
            distance = math.sqrt((x_pos - prev_x)**2 + (y_pos - prev_y)**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            # 평면 경계 확인
            if (abs(x_pos) + sphere_scale <= PLANE_SIZE/2 - SAFE_MARGIN and 
                abs(y_pos) + sphere_scale <= PLANE_SIZE/2 - SAFE_MARGIN):
                break
    
    # 평면 높이 계산
    plane = bpy.data.objects['Plane']
    plane_height = plane.location.z + (plane.dimensions.z / 2)
    
    # 구체를 평면 위에 배치 (구체 바닥이 평면에 닿도록)
    sphere.location = (x_pos, y_pos, plane_height + sphere_scale)
    
    print(f"Chrome ball insertion object added at position: ({x_pos:.2f}, {y_pos:.2f}, {sphere.location.z:.2f})")
    return sphere

def remove_glb_object_for_insertion(glb_object_to_file):
    """이미 씬에 있는 GLB 객체 중 하나를 제거하여 insertion object로 사용"""
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
    target_name = target_object.name  # 제거하기 전에 이름 저장
    target_file = glb_object_to_file.get(target_name, "unknown.glb")  # 파일 경로 가져오기
    
    # 제거하기 전에 객체의 변환 행렬 정보 저장
    target_matrix_world = target_object.matrix_world.copy()  # object-to-world 변환 행렬
    target_location = list(target_object.location)
    target_rotation = list(target_object.rotation_euler)
    target_scale = list(target_object.scale)
    
    print(f"Selected GLB object for removal: {target_name}, file: {target_file}, position: ({target_object.location.x:.2f}, {target_object.location.y:.2f})")
    
    # 선택된 객체와 모든 자식 객체들을 찾아서 제거
    objects_to_remove = []
    
    if target_object.type == 'EMPTY':
        # Empty 객체의 경우 모든 자식 객체들도 수집
        def collect_children(obj):
            objects_to_remove.append(obj)
            for child in obj.children:
                collect_children(child)
        
        collect_children(target_object)
    else:
        # 메시 객체의 경우
        objects_to_remove.append(target_object)
    
    # 객체들을 씬에서 제거
    for obj in objects_to_remove:
        print(f"Removing object: {obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)
    
    print(f"GLB object removed successfully. Total {len(objects_to_remove)} objects removed.")
    return {
        "name": target_name, 
        "file": target_file,
        "matrix_world": [list(row) for row in target_matrix_world],  # object-to-world 변환 행렬
        "location": target_location,
        "rotation": target_rotation,
        "scale": target_scale
    }

def main():
    args = parse_args()
    
    # 작업 디렉토리 설정
    root_dir = Path("/Users/jinwoo/Documents/work/svoi")
    dataset_dir = root_dir / "dataset" / args.dataset_id
    tex_dir = dataset_dir / "textures"
    
    # 디렉토리 존재 확인
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not tex_dir.exists():
        raise FileNotFoundError(f"Texture directory not found: {tex_dir}")
    
    # 상수 설정
    TARGET_MAX_DIM_GLB = args.glb_max_size
    TARGET_MAX_DIM_PRIMITIVE = args.primitive_max_size
    PLANE_SIZE = args.plane_size
    SAFE_MARGIN = args.safe_margin
    
    # GLB 파일 경로 확인
    glb_paths = [
        dataset_dir / "1.glb",
        dataset_dir / "2.glb", 
        dataset_dir / "3.glb"
    ]
    
    for glb_path in glb_paths:
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")
    
    # 씬 초기화
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

    # 텍스처 파일 찾기
    color_path = find_texture_file(tex_dir, args.dataset_id, "diff")
    roughness_path = find_texture_file(tex_dir, args.dataset_id, "rou") 
    normal_path = find_texture_file(tex_dir, args.dataset_id, "nor")

    # 평면 생성
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Plane"

    # 평면 머티리얼 설정
    mat = bpy.data.materials.new(name="Plane_Material")
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

    tex_coord.location = (-800, 0)
    mapping.location = (-600, 0)
    tex_image_color.location = (-400, 200)
    tex_image_rough.location = (-400, 0)
    tex_image_normal.location = (-400, -200)
    normal_map.location = (-200, -200)
    bsdf.location = (0, 0)
    output.location = (200, 0)

    links.new(tex_image_color.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(tex_image_rough.outputs['Color'], bsdf.inputs['Roughness'])
    links.new(tex_image_normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    if len(plane.data.materials) == 0:
        plane.data.materials.append(mat)
    else:
        plane.data.materials[0] = mat

    # 이미지 패킹
    for image in bpy.data.images:
        if not image.packed_file:
            image.pack()

    # GLB 파일들과 primitive 객체들을 섞어서 배치할 리스트 생성
    primitive_types = ['cube', 'sphere', 'cylinder', 'cone', 'torus', 'monkey']

    # GLB 파일들과 primitive 객체들을 랜덤으로 섞기
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

    # 배치된 객체들의 위치를 추적하기 위한 리스트
    placed_positions = []
    
    # GLB 객체와 파일 경로 매핑을 저장하는 딕셔너리
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
            
            # 객체 타입에 따라 다른 크기 적용
            if obj_type == 'glb':
                target_dim = TARGET_MAX_DIM_GLB
            else:  # primitive
                target_dim = TARGET_MAX_DIM_PRIMITIVE
            
            scale_factor = target_dim / current_dim if current_dim > 0 else 1.0
            
            parent_object.scale = (scale_factor, scale_factor, scale_factor)
            
            # 랜덤 Z축 회전 (0~360도)
            random_rotation = random.uniform(0, 2 * math.pi)
            parent_object.rotation_euler = Euler((0, 0, random_rotation), 'XYZ')
            
            # 평면의 정확한 높이 계산 (평면의 중심점 + 평면 높이의 절반)
            plane_height = plane.location.z + (plane.dimensions.z / 2)
            
            # 랜덤 XY 위치 생성 (겹치지 않도록, bounding box가 평면 밖으로 나가지 않도록)
            max_attempts = 30
            for attempt in range(max_attempts):
                # 평면 범위 내에서 랜덤 위치 생성
                x_pos = random.uniform(-PLANE_SIZE/2 + SAFE_MARGIN, PLANE_SIZE/2 - SAFE_MARGIN)
                y_pos = random.uniform(-PLANE_SIZE/2 + SAFE_MARGIN, PLANE_SIZE/2 - SAFE_MARGIN)
                
                # 기존 객체들과 거리 확인
                too_close = False
                current_target = TARGET_MAX_DIM_GLB if obj_type == 'glb' else TARGET_MAX_DIM_PRIMITIVE
                # 실제 스케일된 크기 고려하여 최소 거리 계산
                actual_scale = scale_factor * current_target
                min_distance = actual_scale * 1.5  # 더 여유있는 최소 거리 (1.5배)
                
                for prev_x, prev_y in placed_positions:
                    distance = math.sqrt((x_pos - prev_x)**2 + (y_pos - prev_y)**2)
                    if distance < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    # bounding box가 평면 밖으로 나가지 않는지 확인
                    # 실제 스케일된 크기 고려
                    obj_radius = actual_scale * 0.6  # 더 보수적인 반지름
                    if (abs(x_pos) + obj_radius <= PLANE_SIZE/2 - SAFE_MARGIN and 
                        abs(y_pos) + obj_radius <= PLANE_SIZE/2 - SAFE_MARGIN):
                        break
            
            # 적절한 위치를 못 찾으면 더 안전한 위치 사용
            if attempt == max_attempts - 1:
                x_pos = random.uniform(-PLANE_SIZE/3, PLANE_SIZE/3)
                y_pos = random.uniform(-PLANE_SIZE/3, PLANE_SIZE/3)
            
            placed_positions.append((x_pos, y_pos))
            
            # 먼저 객체를 임시로 배치
            parent_object.location = (x_pos, y_pos, 0)
            
            # 스케일링 후 객체의 실제 바운딩 박스 계산
            bpy.context.view_layer.update()  # 변형 업데이트
            
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
                        # 객체의 밑바닥이 평면에 닿도록 위치 조정
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
            
            print(f"Object {i+1}: type='{obj_type}', name='{parent_object.name}', scale: {scale_factor:.3f}, position: ({x_pos:.2f}, {y_pos:.2f}, {parent_object.location.z:.2f}), rotation: {math.degrees(random_rotation):.1f}°")
        else:
            print(f"Object {i+1}: No parent object found for {obj_type}: {obj_data}")

    # World Environment Map 설정
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

    # Environment Texture 이미지 로드 (동적으로 확장자 찾기)
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

    # 렌더 엔진을 Cycles로 설정
    bpy.context.scene.render.engine = 'CYCLES'

    # 렌더 샘플링 설정
    bpy.context.scene.cycles.samples = args.render_samples
    bpy.context.scene.cycles.preview_samples = args.render_samples

    print(f"Render engine set to Cycles with {args.render_samples} samples")

    # 씬 바운드 계산
    center, size = calculate_scene_bounds()
    print(f"Scene center: {center}, Scene size: {size}")

    # 다양한 대각선 각도 정의
    diagonal_angles = [
        (1.5, 1.5, 1.2),    # 기본 대각선
        (-1.5, 1.5, 1.2),   # 왼쪽 대각선
        (1.5, -1.5, 1.2),   # 오른쪽 대각선
        (-1.5, -1.5, 1.2),  # 뒤쪽 대각선
        (2.0, 1.0, 1.5),    # 높은 각도1
        (-2.0, 1.0, 1.5),   # 높은 각도2
        (1.0, 2.0, 1.5),    # 높은 각도3
        (-1.0, -2.0, 1.5),  # 높은 각도4
        (1.8, 0.8, 0.8),    # 낮은 각도1
        (-1.8, 0.8, 0.8),   # 낮은 각도2
        (0.8, 1.8, 0.8),    # 낮은 각도3
        (-0.8, -1.8, 0.8),  # 낮은 각도4
    ]

    # 랜덤으로 대각선 각도 선택
    selected_angle = random.choice(diagonal_angles)

    # 1. 먼저 기본 씬 렌더링 (insertion object 없음)
    camera = setup_camera_and_render(center, size, "Camera_Diagonal", selected_angle, f"{args.dataset_id}_before.png", dataset_dir)
    print(f"Rendered base scene: {args.dataset_id}_before.png")

    # 2. Insertion object 추가 (사용자 옵션에 따라 chrome ball 또는 GLB 객체)
    insertion_object = None
    
    if args.use_chrome_ball:
        print("Selected insertion type: Chrome Ball")
        insertion_object = add_chrome_ball_insertion_object(placed_positions, PLANE_SIZE, SAFE_MARGIN)
    else:
        print("Selected insertion type: GLB Object (default)")
        removed_object_info = remove_glb_object_for_insertion(glb_object_to_file)
        insertion_object = removed_object_info  # 제거된 객체의 정보를 저장
    
    # 3. Insertion object가 있는 씬 렌더링
    if insertion_object:
        camera = setup_camera_and_render(center, size, "Camera_Diagonal", selected_angle, f"{args.dataset_id}_after.png", dataset_dir)
        if args.use_chrome_ball:
            print(f"Rendered scene with Chrome Ball: {args.dataset_id}_after.png")
        else:
            print(f"Rendered scene with GLB Object removed ({removed_object_info['name']}): {args.dataset_id}_after.png")
    else:
        print("Warning: Failed to add insertion object")

    print(f"Selected diagonal angle: {selected_angle}")
    print("Object insertion dataset completed!")

    # 카메라와 객체 정보를 메타데이터로 저장
    import json
    
    # 카메라의 월드 행렬 (T_C_to_A) 추출
    T_C_to_A = camera.matrix_world
    T_A_to_C = T_C_to_A.inverted()  # 뷰 행렬
    
    metadata = {
        "camera_location": list(camera.location),
        "camera_rotation": list(camera.rotation_euler),
        "camera_matrix_world": [list(row) for row in T_C_to_A],  # T_C_to_A (4x4 행렬)
        "camera_view_matrix": [list(row) for row in T_A_to_C],   # T_A_to_C (뷰 행렬)
        "scene_center": list(center),
        "selected_angle": selected_angle,
        "insertion_object": None
    }
    
    # 제거된 객체 정보 저장 (GLB 객체인 경우)
    if not args.use_chrome_ball and insertion_object:
        metadata["insertion_object"] = {
            "name": insertion_object["name"],
            "file": insertion_object["file"],
            "type": "glb_object",
            "matrix_world": insertion_object["matrix_world"],  # object-to-world 변환 행렬
            "location": insertion_object["location"],
            "rotation": insertion_object["rotation"],
            "scale": insertion_object["scale"]
        }
        print(f"Saved removed object info: {insertion_object['name']} (file: {insertion_object['file']}) with transform matrix")
    elif args.use_chrome_ball and insertion_object:
        # Chrome ball의 경우
        metadata["insertion_object"] = {
            "type": "chrome_ball",
            "location": list(insertion_object.location),
            "scale": list(insertion_object.scale)
        }
        print("Saved chrome ball info")
    
    metadata_path = dataset_dir / f"{args.dataset_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    # 블렌더 파일 저장
    blend_path = dataset_dir / f"{args.dataset_id}.blend"
    if blend_path.exists():
        blend_path.unlink()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

if __name__ == "__main__":
    main()
