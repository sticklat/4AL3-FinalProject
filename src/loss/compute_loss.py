import argparse, pandas as pd, numpy as np, json, sys
from pathlib import Path

def mae(a, b): return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann",  required=True, help="Ground truth CSV (has z_true or zloc)")
    ap.add_argument("--pred", required=True, help="Predictions CSV (must have z_pred)")
    ap.add_argument("--out",  required=True, help="Output metrics CSV")
    ap.add_argument("--key",  default="",   help="Optional join key present in both files (e.g., row_id)")
    args = ap.parse_args()

    ann  = pd.read_csv(args.ann)
    pred = pd.read_csv(args.pred)

    # Normalize GT distance column
    if "z_true" not in ann.columns:
        if "zloc" in ann.columns: ann = ann.rename(columns={"zloc":"z_true"})
        elif "z" in ann.columns:  ann = ann.rename(columns={"z":"z_true"})
        else: sys.exit("No z_true/zloc/z in annotations file.")

    # Choose join columns
    if args.key and args.key in ann.columns and args.key in pred.columns:
        join_cols = [args.key]
    else:
        needed = ["filename","xmin","ymin","xmax","ymax"]
        if not all(c in ann.columns for c in needed) or not all(c in pred.columns for c in needed):
            sys.exit("Provide --key present in both files, or include filename,xmin,ymin,xmax,ymax in both.")
        join_cols = needed

    merged = ann.merge(pred, on=join_cols, how="inner", suffixes=("_ann","_pred"))
    if "z_pred" not in merged.columns:
        sys.exit("Predictions CSV must contain z_pred.")

    y    = merged["z_true"].to_numpy(dtype=float)
    yhat = merged["z_pred"].to_numpy(dtype=float)

    metrics = {"n": int(len(merged)), "mae": mae(y, yhat), "rmse": rmse(y, yhat)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.out, index=False)
    merged.to_csv(Path(args.out).with_suffix(".detailed.csv"), index=False)

    print("✅ Saved", args.out)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
