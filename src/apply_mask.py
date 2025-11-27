import argparse
import os
import glob
from typing import Optional

import numpy as np
from PIL import Image


def apply_mask_to_image(
    image_path: str,
    mask_path: str,
    output_path: Optional[str] = None,
    threshold: int = 0,
    invert: bool = False,
    transparent_background: bool = True,
    resize_mask: bool = True,
) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    try:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
    except Exception as exc:
        raise RuntimeError("Failed to open image or mask") from exc

    if mask.mode not in ["RGBA", "LA"]:
        mask = mask.convert("RGBA")

    if image.size != mask.size:
        if resize_mask:
            mask = mask.resize(image.size, Image.NEAREST)
        else:
            raise ValueError(
                f"Image and mask sizes differ: image={image.size}, mask={mask.size}."
            )

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    if mask.mode == "RGBA":
        mask_alpha = np.array(mask)[..., 3]
    else:
        mask_alpha = np.array(mask)[..., 1]

    keep = (mask_alpha > np.uint8(threshold))
    if invert:
        keep = ~keep

    alpha = np.where(keep, 255, 0).astype(np.uint8)
    rgba = np.array(image)

    if transparent_background:
        rgba[..., 3] = alpha
    else:
        keep_u8 = keep.astype(np.uint8)
        rgba[..., 0] = rgba[..., 0] * keep_u8
        rgba[..., 1] = rgba[..., 1] * keep_u8
        rgba[..., 2] = rgba[..., 2] * keep_u8
        rgba[..., 3] = 255

    result = Image.fromarray(rgba)

    if output_path is None:
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, "result_object3.png")

    try:
        result.save(output_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save masked image: {output_path}") from exc

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이미지에 마스킹 적용")
    parser.add_argument("experiment_num", help="실험 번호")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_num = args.experiment_num
    
    mask_path = f"dataset/{experiment_num}/{experiment_num}_object.png"
    out_dir = f"out/{experiment_num}"
    
    if not os.path.isfile(mask_path):
        print(f"마스크 파일을 찾을 수 없습니다: {mask_path}")
        return
    
    if not os.path.isdir(out_dir):
        print(f"출력 디렉토리를 찾을 수 없습니다: {out_dir}")
        return
    
    png_files = glob.glob(os.path.join(out_dir, "*.png"))
    target_files = [f for f in png_files if not os.path.basename(f).endswith("_object.png")]
    
    if not target_files:
        print(f"{out_dir}에서 마스킹할 PNG 파일을 찾을 수 없습니다.")
        return
    
    print(f"실험 번호: {experiment_num}")
    print(f"마스크 파일: {mask_path}")
    print(f"마스킹할 파일 개수: {len(target_files)}")
    
    for image_path in target_files:
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}_object.png"
        output_path = os.path.join(out_dir, output_filename)
        
        try:
            apply_mask_to_image(
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_path,
                threshold=0,
                invert=False,
                transparent_background=True,
                resize_mask=True,
            )
            print(f"완료: {filename} -> {output_filename}")
        except Exception as e:
            print(f"오류 발생 {filename}: {e}")


if __name__ == "__main__":
    main()