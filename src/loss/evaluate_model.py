# src/Loss/evaluate_model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from estimator.NN import DistanceRegressor  

# ---- Load dataset (force CPU) ----
ds = torch.load("src/estimator/dataset/vehicle_dataset.pt",
                map_location=torch.device("cpu"))
X = ds["features"]              # tensor [N, D]
y_true_t = ds["distances"]      # tensor [N] or [N,1]

# ---- Build model (match teammate’s config) ----
model = DistanceRegressor(input_dim=X.shape[1],
                          hidden_dim=128,
                          num_hidden_layers=6,
                          activation=nn.ELU)

# ---- Load trained weights (force CPU) ----
state = torch.load("src/estimator/epochs300_layers6_dim128_actELU_lr5e-04.pt",
                   map_location=torch.device("cpu"))
# allow strict or not depending on exact keys
model.load_state_dict(state, strict=False)
model.eval()

# ---- Predict ----
with torch.no_grad():
    y_pred_t = model(X)

# ---- Ensure flat 1-D numpy arrays ----
y_pred = y_pred_t.detach().cpu().numpy().reshape(-1)
y_true = y_true_t.detach().cpu().numpy().reshape(-1)

# ---- Export predictions with a stable key ----
row_id = np.arange(len(y_pred))
df_pred = pd.DataFrame({"row_id": row_id, "z_pred": y_pred})
df_pred.to_csv("reports/model_preds.csv", index=False)
print("✅ Saved predictions to reports/model_preds.csv")
print(df_pred.head())
