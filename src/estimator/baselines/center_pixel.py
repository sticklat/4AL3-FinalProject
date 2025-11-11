import pandas as pd
from pathlib import Path
from typing import Optional
from estimator.utils import load_grayscale_png, center_pixel_depth

def run_center_pixel(ann_csv: str, depth_root: str, depth_subdir="proj_depth/groundtruth/image_02",
                     filename_col="filename", x1="xmin", y1="ymin", x2="xmax", y2="ymax",
                     class_col="class") -> pd.DataFrame:
    """
    Returns a DataFrame with columns: filename, class, z_true, z_pred
    Assumes `ann_csv` has bbox + true z (if available). If z_true not in CSV,
    you can compute it similarly with bbox_median_depth.
    """
    df = pd.read_csv(ann_csv)
    has_true = "z" in df.columns or "z_true" in df.columns
    ztrue_col = "z_true" if "z_true" in df.columns else ("z" if "z" in df.columns else None)

    rows = []
    for _, r in df.iterrows():
        stem = Path(r[filename_col]).stem  # e.g., 003871.txt -> 003871
        depth_path = Path(depth_root) / depth_subdir / f"{stem}.png"
        dm = load_grayscale_png(str(depth_path))
        box = (r[x1], r[y1], r[x2], r[y2])
        z_pred = center_pixel_depth(dm, box)
        if z_pred is None:
            continue
        row = {
            "filename": r[filename_col],
            "class": r.get(class_col, "Unknown"),
            "z_pred": z_pred,
        }
        if has_true:
            row["z_true"] = r[ztrue_col]
        rows.append(row)
    return pd.DataFrame(rows)