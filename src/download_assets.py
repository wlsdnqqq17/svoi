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
    random_uids = random.sample(all_uids, 3)

    print(f"Downloading 3 random objects: {random_uids}")
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


if __name__ == "__main__":
    main()
