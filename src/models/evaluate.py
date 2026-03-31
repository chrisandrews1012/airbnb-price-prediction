""" 
Model Evaluation pipeline.
Computes performance metrics on the held-out test set. 

Metrics are computed on the original price scale (not log-scale) so they are interpretable to stakeholders.
Results are saved `models/model_metadata.json` for display in the Streamlit Model Insights page.
"""

import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_path: str = "models/model_metadata.json") -> dict:
    """
    Evaluate the trained pipeline on the held-out test set.

    Predictions are converted from log-price back to £ before computing metrics
    so all values are interpretable in real terms.

    :param pipeline: Fitted sklearn Pipeline.
    :type pipeline: Pipeline
    :param X_test: Test features.
    :type X_test: pd.DataFrame
    :param y_test: True log-price values for the test set.
    :type y_test: pd.Series
    :param output_path: Path to save the metrics JSON.
    :type output_path: str

    :returns: Dictionary of evaluation metrics.
    :rtype: dict
    """
    # Predict and convert back from log-scale to original price scale (£)
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # inverse of log1p
    y_true = np.expm1(y_test)  # inverse of log1p
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    median_ae = np.median(np.abs(y_true - y_pred))
    
    metrics = {
        "rmse":       round(rmse, 2),
        "mae":        round(mae, 2),
        "mape":       round(mape, 2),
        "r2":         round(r2, 4),
        "median_ae":  round(median_ae, 2),
        "test_size":  len(y_test),
    }
    
    print("\nTest Set Evaluation (£ scale):")
    print(f"  RMSE:             £{rmse:.2f}")
    print(f"  MAE:              £{mae:.2f}")
    print(f"  MAPE:             {mape:.2f}%")
    print(f"  R²:               {r2:.4f}")
    print(f"  Median Abs Error: £{median_ae:.2f}")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {output_path}")

    return metrics