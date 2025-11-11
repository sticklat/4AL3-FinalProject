import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from estimator.metrics import summarize
from estimator.baselines.center_pixel import run_center_pixel
from estimator.baselines.box_median_oracle import run_box_median
from estimator.utils import load_grayscale_png

def _crop(image_path: Path, box, size=224):
    from torchvision.io import read_image
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    img = Image.open(image_path).convert("RGB")
    img = img.crop((x1,y1,x2,y2)).resize((size,size))
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return t(img).unsqueeze(0)  # 1x3xHxW

def predict_with_model(ann_csv: str, img_root: str, model_path: str,
                       filename_col="filename", img_subdir="image_02/data",
                       x1="xmin", y1="ymin", x2="xmax", y2="ymax",
                       class_col="class") -> pd.DataFrame:
    """
    Generic inference for a crop-based scalar regressor saved as .pt (state_dict or scripted).
    Expects input: 1x3x224x224 -> output: 1x1 (meters).
    """
    df = pd.read_csv(ann_csv)
    rows = []
    # Load model (supports both scripted and Python module checkpoints)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        scripted = True
    except Exception:
        scripted = False
        sd = torch.load(model_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        # Minimal fallback model definition (adjust to your architecture if needed)
        import torch.nn as nn
        class TinyRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.head = nn.Linear(64, 1)
            def forward(self, x):
                x = self.backbone(x).flatten(1)
                return self.head(x).squeeze(1)
        model = TinyRegressor()
        model.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict())
        model.eval()

    for _, r in df.iterrows():
        stem = Path(r[filename_col]).stem  # 003871.txt -> 003871
        img_path = Path(img_root) / img_subdir / f"{stem}.png"
        box = (r[x1], r[y1], r[x2], r[y2])
        x = _crop(img_path, box, size=224)
        with torch.no_grad():
            yhat = model(x)
            if yhat.ndim > 1:  # handle (1,1) or similar
                yhat = yhat.squeeze()
            z_pred = float(yhat.item())
        rows.append({
            "filename": r[filename_col],
            "class": r.get(class_col, "Unknown"),
            "z_pred": z_pred,
            "z_true": r.get("z_true", r.get("z", None))
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="annotations CSV with filename, class, bbox, and optional z_true")
    ap.add_argument("--depth-root", required=False, default="", help="KITTI root for depth (for baselines)")
    ap.add_argument("--img-root", required=False, default="", help="KITTI root for RGB images (for model mode)")
    ap.add_argument("--pred-mode", required=True, choices=["box_median", "center_pixel", "model"])
    ap.add_argument("--model", required=False, help="path to .pt regressor (if pred-mode=model)")
    ap.add_argument("--out", required=True, help="output CSV summary path (we also save detailed preds next to it)")
    args = ap.parse_args()

    if args.pred_mode == "center_pixel":
        df = run_center_pixel(args.ann, depth_root=args.depth_root)
    elif args.pred_mode == "box_median":
        df = run_box_median(args.ann, depth_root=args.depth_root)
    else:
        if not args.model or not args.img_root:
            raise SystemExit("--model and --img-root are required when --pred-mode=model")
        df = predict_with_model(args.ann, img_root=args.img_root, model_path=args.model)

    # drop NaNs if any
    df = df.dropna(subset=["z_pred"])
    # save detailed predictions
    out_detail = Path(args.out).with_suffix(".detailed.csv")
    df.to_csv(out_detail, index=False)

    # if we have ground truth, summarize
    if "z_true" in df.columns and df["z_true"].notna().any():
        summary = summarize(df, y_col="z_true", yhat_col="z_pred", class_col="class")
        # write a one-line CSV summary and a per-class/per-bin CSV
        import json
        summary_row = {
            "overall_mae": summary.overall_mae,
            "overall_rmse": summary.overall_rmse,
            "mae_by_class": json.dumps(summary.mae_by_class),
            "mae_by_distbin": json.dumps(summary.mae_by_distbin),
        }
        pd.DataFrame([summary_row]).to_csv(args.out, index=False)
        print("Saved:", args.out, "and", out_detail)
    else:
        # no ground truth—just save predictions
        pd.DataFrame([{"note": "no z_true in input; saved predictions only"}]).to_csv(args.out, index=False)
        print("Saved predictions:", out_detail)

if __name__ == "__main__":
    main()