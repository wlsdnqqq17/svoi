import glob
import json
import os

import pandas as pd

base_pattern = "./out/{idx}/eval/*.json"
start_idx = 201
end_idx = 210

rows = []
max_images = 0
for idx in range(start_idx, end_idx + 1):
    pattern = base_pattern.format(idx=idx)
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        continue

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read {jf}: {e}")
            continue

        folder_name = os.path.basename(os.path.dirname(os.path.dirname(jf)))
        file_name = os.path.basename(jf)
        row = {"file": f"{folder_name}_{file_name}"}

        for i, item in enumerate(data, start=1):
            psnr = item.get("psnr") if isinstance(item, dict) else None
            ssim = item.get("ssim") if isinstance(item, dict) else None
            lpips = item.get("lpips") if isinstance(item, dict) else None

            row[f"psnr_{i}"] = psnr
            row[f"ssim_{i}"] = ssim
            row[f"lpips_{i}"] = lpips

        if len(data) > max_images:
            max_images = len(data)

        rows.append(row)

cols = ["file"]
for i in range(1, max_images + 1):
    cols += [f"psnr_{i}", f"ssim_{i}", f"lpips_{i}"]

df = pd.DataFrame(rows, columns=cols)

psnr_cols = [c for c in df.columns if c.startswith("psnr_")]
ssim_cols = [c for c in df.columns if c.startswith("ssim_")]
lpips_cols = [c for c in df.columns if c.startswith("lpips_")]

out_csv = "metrics_table.csv"
df.to_csv(out_csv, index=False)
print(f"saved: {out_csv}")
