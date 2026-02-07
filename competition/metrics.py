# metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_submission(y_true, y_pred, is_cold_start=None):
    """
    Evaluate MovieLens predictions
    
    Args:
        y_true: Ground truth ratings (1-5)
        y_pred: Predicted ratings
        is_cold_start: Optional boolean array indicating cold start instances
    
    Returns:
        dict with metrics
    """
    # Clip predictions to valid range
    y_pred_clipped = np.clip(y_pred, 1.0, 5.0)
    
    # Overall RMSE
    rmse_overall = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    
    results = {
        'rmse': float(rmse_overall),
        'primary_metric': float(rmse_overall)
    }
    
    # Cold-start breakdown (if provided)
    if is_cold_start is not None:
        cold_mask = is_cold_start.astype(bool)
        warm_mask = ~cold_mask
        
        if cold_mask.sum() > 0:
            rmse_cold = np.sqrt(mean_squared_error(
                y_true[cold_mask],
                y_pred_clipped[cold_mask]
            ))
            results['rmse_cold_start'] = float(rmse_cold)
        
        if warm_mask.sum() > 0:
            rmse_warm = np.sqrt(mean_squared_error(
                y_true[warm_mask],
                y_pred_clipped[warm_mask]
            ))
            results['rmse_warm_start'] = float(rmse_warm)
    
    return results

def compute_score(y_true, y_pred):
    """Backward compatibility: return primary metric only"""
    return evaluate_submission(y_true, y_pred)['primary_metric']