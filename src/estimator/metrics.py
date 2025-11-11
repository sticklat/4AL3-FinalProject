from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

@dataclass
class EvalSummary:
    overall_mae: float
    overall_rmse: float
    mae_by_class: Dict[str, float] = field(default_factory=dict)
    mae_by_distbin: Dict[str, float] = field(default_factory=dict)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true))) if len(y_true) else float("nan")

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if len(y_true) else float("nan")

def distance_bins(z: np.ndarray) -> List[str]:
    # bins: [0–10), [10–30), [30–80], tweak as needed
    labels = []
    for v in z:
        if v < 10: labels.append("[0,10)")
        elif v < 30: labels.append("[10,30)")
        else: labels.append("[30,80]")
    return labels

def summarize(df: pd.DataFrame, y_col="z_true", yhat_col="z_pred", class_col="class") -> EvalSummary:
    y = df[y_col].to_numpy()
    yhat = df[yhat_col].to_numpy()
    overall = EvalSummary(overall_mae=_mae(y, yhat), overall_rmse=_rmse(y, yhat))

    # per-class
    if class_col in df.columns:
        for c, sub in df.groupby(class_col):
            overall.mae_by_class[c] = _mae(sub[y_col].to_numpy(), sub[yhat_col].to_numpy())

    # by distance bin
    bins = distance_bins(y)
    df_bins = df.copy()
    df_bins["dist_bin"] = bins
    for b, sub in df_bins.groupby("dist_bin"):
        overall.mae_by_distbin[b] = _mae(sub[y_col].to_numpy(), sub[yhat_col].to_numpy())
    return overall