import objaverse
import random
import os
import shutil
import gzip
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download assets from Objaverse")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID")
    args = parser.parse_args()

    root_dir = Path("/Users/jinwoo/Documents/work/svoi")
    download_dir = root_dir / "dataset" / args.dataset_id
    raw_dir = root_dir / "dataset_legacy" / "hf-objaverse-v1"
    os.makedirs(download_dir, exist_ok=True)
    objaverse.BASE_PATH = download_dir
    objaverse._VERSIONED_PATH = raw_dir

    lvis_annotations = objaverse.load_lvis_annotations()
    all_uids = [uid for uids in lvis_annotations.values() for uid in uids]
    random_uids = random.sample(all_uids, 2)

    print(f"Downloading 2 random objects: {random_uids}")
    objaverse.load_objects(uids=random_uids, download_processes=1)

    object_paths_path = raw_dir / "object-paths.json.gz"
    with gzip.open(object_paths_path, "rt") as f:
        path_map = json.load(f)

    for i, uid in enumerate(random_uids, start=1):
        rel_path = path_map.get(uid)
        if rel_path is None:
            print(f"[ERROR] UID {uid} not found in object-paths.json.gz")
            continue

        src = raw_dir / rel_path
        dst = download_dir / f"{i}.glb"

        if src.exists():
            shutil.copyfile(src, dst)
            print(f"[OK] Copied {uid} → {dst.name}")
        else:
            print(f"[ERROR] File for UID {uid} not found at {src}")

    # For 3.glb, pick a random local GLB from svoi/objs
    local_objs_dir = root_dir / "objs"
    if local_objs_dir.exists():
        local_glbs = [p for p in local_objs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".glb"]
        if local_glbs:
            chosen_glb = random.choice(local_glbs)
            dst_third = download_dir / "3.glb"
            shutil.copyfile(chosen_glb, dst_third)
            print(f"[OK] Copied local GLB {chosen_glb.name} → 3.glb")
        else:
            print("[ERROR] No .glb files found in svoi/objs directory")
    else:
        print("[ERROR] svoi/objs directory not found")

    # Copy random HDRI as envmap.exr
    hdri_dir = root_dir / "HDRIs"
    if hdri_dir.exists():
        hdri_files = list(hdri_dir.glob("*.exr"))
        if hdri_files:
            random_hdri = random.choice(hdri_files)
            envmap_dst = download_dir / "envmap.exr"
            shutil.copyfile(random_hdri, envmap_dst)
            print(f"[OK] Copied random HDRI {random_hdri.name} → envmap.exr")
        else:
            print("[ERROR] No .exr files found in HDRIs directory")
    else:
        print("[ERROR] HDRIs directory not found")

    # Copy random texture folder as textures
    textures_root_dir = root_dir / "textures"
    if textures_root_dir.exists():
        texture_folders = [d for d in textures_root_dir.iterdir() if d.is_dir()]
        if texture_folders:
            random_texture_folder = random.choice(texture_folders)
            textures_dst = download_dir / "textures"
            if textures_dst.exists():
                shutil.rmtree(textures_dst)
            shutil.copytree(random_texture_folder, textures_dst)
            print(f"[OK] Copied random texture folder {random_texture_folder.name} → textures/")
        else:
            print("[ERROR] No texture folders found in textures directory")
    else:
        print("[ERROR] textures directory not found")


if __name__ == "__main__":
    main()
