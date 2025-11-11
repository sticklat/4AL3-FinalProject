# 4AL3-FinalProject

# Model Evaluation & Loss Computation
## 🔧 Environment Setup

Make sure the required dependencies are installed:
```bash
pip install torch pandas numpy
```

Then set the Python path to the project’s `src` directory:
```bash
export PYTHONPATH=$PWD/src
```

---

## 🚀 Step 1: Run Model Evaluation

This step loads the trained model (`.pt` file) and generates predictions from the dataset.

```bash
python3 src/Loss/evaluate_model.py
```

If everything works correctly, it will produce:
```
reports/model_preds.csv
```

Example output:
```
✅ Saved predictions to reports/model_preds.csv
   row_id     z_pred
0       0    8.281875
1       1   73.413963
2       2   58.318951
3       3   46.285900
4       4   10.187097
```

---

## 📊 Step 2: Compute Loss Metrics

Now compare predictions with ground-truth data and compute evaluation metrics (MAE, RMSE, etc.):  

```bash
python3 src/Loss/compute_loss.py \
  --ann src/loss/data/annotations_with_id.csv \
  --pred reports/model_preds.csv \
  --out reports/metrics_model.csv \
  --key row_id
```

Expected output:
```
✅ Saved reports/metrics_model.csv
{
  "n": 40570,
  "mae": 1.52,
  "rmse": 2.17
}
```

---

## 📁 Output Files

After running both scripts, the following files will be created under `reports/`:
```
reports/model_preds.csv
reports/metrics_model.csv
reports/metrics_model.detailed.csv
```

---