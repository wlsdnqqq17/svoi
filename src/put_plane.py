import bpy
import os
import random
import sys
import math
from mathutils import Vector, Euler

if len(sys.argv) > 1:
    dir_name = sys.argv[1]
else:
    dir_name = "001"

tex_dir = f"dataset/{dir_name}/abstract_1-4K" 
name = "4K-abstract_1"

for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

color_path = os.path.join(tex_dir, f"{name}-diffuse.jpg")
roughness_path = os.path.join(tex_dir, f"{name}-specular.jpg")
normal_path = os.path.join(tex_dir, f"{name}-normal.jpg")

bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = "AbstractPlane"
plane_size = plane.dimensions.xy 


mat = bpy.data.materials.new(name="Abstract_Material")
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


for image in bpy.data.images:
    if not image.packed_file:
        image.pack()

# Blender 내장 primitive 객체 생성 함수
def create_primitive_object(primitive_type, location=(0, 0, 0)):
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
    import colorsys
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
    
    return obj

# GLB 파일들과 primitive 객체들을 섞어서 배치할 리스트 생성
glb_paths = [
    f"dataset/{dir_name}/1.glb",
    f"dataset/{dir_name}/2.glb",
    f"dataset/{dir_name}/3.glb"
]

primitive_types = ['cube', 'sphere', 'cylinder', 'cone', 'torus', 'monkey']

# GLB 파일들과 primitive 객체들을 랜덤으로 섞기
all_objects = []
# 모든 GLB 파일 추가
for glb_path in glb_paths:
    all_objects.append(('glb', glb_path))

# 3-5개의 랜덤 primitive 객체 추가
num_primitives = random.randint(3, 5)
for _ in range(num_primitives):
    primitive_type = random.choice(primitive_types)
    all_objects.append(('primitive', primitive_type))

# 리스트 섞기
random.shuffle(all_objects)

TARGET_MAX_DIM = 1.0
PLANE_SIZE = 5.0  # 평면 크기
SAFE_MARGIN = 0.3  # 평면 가장자리에서 안전 여백

# 배치된 객체들의 위치를 추적하기 위한 리스트
placed_positions = []

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
    
    elif obj_type == 'primitive':
        # Primitive 객체 생성
        parent_object = create_primitive_object(obj_data, location=(0, 0, 0))
        parent_object.name = f"{obj_data}_{i}"
    
    if parent_object:
        if parent_object.type == 'EMPTY':
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            def get_all_mesh_children(obj):
                meshes = []
                if obj.type == 'MESH':
                    meshes.append(obj)
                for child in obj.children:
                    meshes.extend(get_all_mesh_children(child))
                return meshes
            
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
        
        scale_factor = TARGET_MAX_DIM / current_dim if current_dim > 0 else 1.0
        
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
            min_distance = TARGET_MAX_DIM * 0.8  # 최소 거리 증가
            
            for prev_x, prev_y in placed_positions:
                distance = math.sqrt((x_pos - prev_x)**2 + (y_pos - prev_y)**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                # bounding box가 평면 밖으로 나가지 않는지 확인
                # 객체의 스케일된 크기 고려
                obj_radius = TARGET_MAX_DIM * 0.5  # 객체의 반지름 (대략적)
                if (abs(x_pos) + obj_radius <= PLANE_SIZE/2 - SAFE_MARGIN and 
                    abs(y_pos) + obj_radius <= PLANE_SIZE/2 - SAFE_MARGIN):
                    break
            
            too_close = False  # 다음 시도 준비
        
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

# Environment Texture 이미지 로드 (HDR 파일)
env_image_path = f"dataset/{dir_name}/envmap.hdr"
if os.path.exists(env_image_path):
    env_tex.image = bpy.data.images.load(env_image_path)
    print(f"World environment map loaded: {env_image_path}")
    # Environment Texture을 World Output에 연결
    world_links.new(env_tex.outputs['Color'], world_output.inputs['Surface'])
    
    # 환경맵 이미지 패킹
    if env_tex.image and not env_tex.image.packed_file:
        env_tex.image.pack()
        print(f"Environment map packed: {env_image_path}")
else:
    raise FileNotFoundError(f"Environment map not found: {env_image_path}")

# World 강도 조정 (현재 Blender 버전에서는 기본 설정 사용)

# 렌더 엔진을 Cycles로 설정
bpy.context.scene.render.engine = 'CYCLES'

# 뷰포트 샘플링 설정
bpy.context.scene.cycles.preview_samples = 64

print("Render engine set to Cycles with viewport samples: 64")

blend_path = os.path.join(f"dataset/{dir_name}", f"{dir_name}.blend")
if os.path.exists(blend_path):
    os.remove(blend_path)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)