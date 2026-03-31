""" 
Model Benchmarking - compares multiple candidate models using 5-fold cross-validation.

Run this once offline to justify the model selection decision.
Results are saved to models/benchmark_results.json and displayed in the app.
"""

import json
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor

from src.models.train import BINARY_COLS, NUMERIC_COLS, OHE_COLS, TARGET_ENCODE_COLS, build_pipeline

CANDIDATE_MODELS = {
    "Ridge":         Ridge(),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "XGBoost":       XGBRegressor(random_state=42, verbosity=0),
    "LightGBM":      LGBMRegressor(random_state=42, verbose=-1),
}

def run_benchmark(df: pd.DataFrame, output_path: str = "models/benchmark_results.json") -> pd.DataFrame:
    """
    Run all candidate models through the same pipeline and compare their performance using 5-fold cross-validation.
    
    :param df: Fully engineered DataFrame from `build_features()`.
    :type df: pd.DataFrame
    :param output_path: File path to save the benchmark results JSON.
    :type output_path: str
    
    :returns: DataFrame of results sorted by RMSE ascending.
    :rtype: pd.DataFrame
    """
    X = df[NUMERIC_COLS + BINARY_COLS + OHE_COLS + TARGET_ENCODE_COLS]
    y = df['log_price']
    
    results = []
    
    for name, model in CANDIDATE_MODELS.items():
        print(f"Benchmarking {name}...")
        
        pipeline = build_pipeline()
        pipeline.set_params(model=model)
        
        # 5-fold CV (metrics are on log_price scale, consistent across all models)
        cv_results = cross_validate(
            pipeline, X, y,
            cv=5,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
            n_jobs=-1,
        )
        
        rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
        mae = -cv_results['test_neg_mean_absolute_error'].mean()
        r2 = cv_results['test_r2'].mean()
        
        results.append({
            "model": name,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
        })
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    
    print(f"\nWinner: {results_df.iloc[0]['model']} with RMSE {results_df.iloc[0]['rmse']:.4f}")
    
    # Save results to JSON for display in the app
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {output_path}")
    
    return results_df