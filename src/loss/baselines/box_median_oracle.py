import pandas as pd
from pathlib import Path
from estimator.utils import load_grayscale_png, bbox_median_depth

def run_box_median(ann_csv: str, depth_root: str, depth_subdir="proj_depth/groundtruth/image_02",
                   filename_col="filename", x1="xmin", y1="ymin", x2="xmax", y2="ymax",
                   class_col="class") -> pd.DataFrame:
    """
    Uses the median of valid depth pixels inside each bbox as prediction.
    This acts as an 'oracle' when your training label is the same median.
    """
    df = pd.read_csv(ann_csv)
    rows = []
    for _, r in df.iterrows():
        stem = Path(r[filename_col]).stem  # 003871.txt -> 003871
        depth_path = Path(depth_root) / depth_subdir / f"{stem}.png"
        dm = load_grayscale_png(str(depth_path))
        box = (r[x1], r[y1], r[x2], r[y2])
        z_pred = bbox_median_depth(dm, box)
        if z_pred is None:
            continue
        rows.append({
            "filename": r[filename_col],
            "class": r.get(class_col, "Unknown"),
            "z_pred": z_pred,
            "z_true": r.get("z_true", r.get("z", None))  # use if present
        })
    return pd.DataFrame(rows)