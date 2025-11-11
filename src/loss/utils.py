from typing import Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

def load_grayscale_png(path: str) -> np.ndarray:
    """Loads PNG as float32 (no scaling). For KITTI depth maps where pixel value = depth (m)."""
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    return arr.astype(np.float32)

def bbox_median_depth(depth_map: np.ndarray, box: Tuple[int,int,int,int], min_valid_px: int = 50) -> Optional[float]:
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1,0), max(y1,0)
    x2, y2 = min(x2, depth_map.shape[1]-1), min(y2, depth_map.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_map[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > 0)
    vals = patch[valid]
    if vals.size < min_valid_px:
        return None
    return float(np.median(vals))

def center_pixel_depth(depth_map: np.ndarray, box: Tuple[int,int,int,int]) -> Optional[float]:
    x1,y1,x2,y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
        v = float(depth_map[cy, cx])
        return v if np.isfinite(v) and v > 0 else None
    return None
