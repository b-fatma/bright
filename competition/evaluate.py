# evaluate.py
"""
Evaluate submission against ground truth labels
"""
import pandas as pd
import sys
from metrics import evaluate_submission

def main(pred_path, label_path):
    # Load files
    preds = pd.read_csv(pred_path).sort_values("id")
    labels = pd.read_csv(label_path).sort_values("id")
    
    # Merge on id
    merged = labels.merge(preds, on="id", how="inner")
    if len(merged) != len(labels):
        raise ValueError("ID mismatch between predictions and labels")
    
    # Extract data
    y_true = merged["rating"].values
    y_pred = merged["y_pred"].values
    is_cold_start = merged["is_cold_start"].values if "is_cold_start" in merged.columns else None
    
    # Evaluate
    results = evaluate_submission(y_true, y_pred, is_cold_start)
    
    # Print results
    print(f"SCORE={results['primary_metric']:.8f}")
    print(f"RMSE={results['rmse']:.8f}")
    
    if 'rmse_cold_start' in results:
        print(f"RMSE_COLD={results['rmse_cold_start']:.8f}")
    if 'rmse_warm_start' in results:
        print(f"RMSE_WARM={results['rmse_warm_start']:.8f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <predictions.csv> <test_labels.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
