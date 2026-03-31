""" 
Training Script -> Run this once to produce all model artifacts 

Usage:
    uv run python scripts/train_model.py
    
Outputs:
    models/pipeline.joblib                      - The fitted sklearn Pipeline (preprocessing + model)
    models/shap_explainer.joblib                - The fitted SHAP TreeExplainer for generating explanations in the API
    models/model_metadata.json                  - Evaluation metrics on the held-out test set
    models/benchmark_metrics.json               - Evaluation metrics for model comparison 
    data/processed/listings_clean.parquet       - The processed dataset with engineered features, saved in efficient Parquet format for future use
"""

import joblib
import shap

from src.data.preprocessor import build_features, save_processed
from src.models.benchmark import run_benchmark
from src.models.evaluate import evaluate
from src.models.train import train, NUMERIC_COLS, BINARY_COLS, OHE_COLS, TARGET_ENCODE_COLS

RAW_DATA_PATH       = "data/listings.csv"
PROCESSED_DATA_PATH = "data/processed/listings_clean.parquet"
PIPELINE_PATH       = "models/pipeline.joblib"
EXPLAINER_PATH      = "models/shap_explainer.joblib"
METADATA_PATH       = "models/model_metadata.json"
BENCHMARK_PATH      = "models/benchmark_results.json"

if __name__ == "__main__":
    
    # Step 1: Build and save processed dataset with engineered features
    print("Step 1: Preprocessing data and engineering features...")
    df = build_features(RAW_DATA_PATH)
    save_processed(df, PROCESSED_DATA_PATH)
    print(f"Processed dataset saved — shape: {df.shape}\n")
    
    # Step 2: Benchmark candidate models 
    print("Step 2: Benchmarking candidate models...")
    benchmark_results = run_benchmark(df, output_path=BENCHMARK_PATH)
    
    # Step 3: Train the final model with Optuna hyperparameter tuning
    print("\nStep 3: Training final LightGBM model with Optuna...")
    pipeline, X_test, y_test = train(df)
    
    # Step 4: Evaluate the final model on the held-out test set and save metrics
    print("\nStep 4: Evaluating final model on test set...")
    evaluate(pipeline, X_test, y_test, output_path=METADATA_PATH)
    
    # Step 5: Save the pipeline
    print("\nStep 5: Saving the trained pipeline...")
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Pipeline saved to {PIPELINE_PATH}")
    
    # Step 6: Build and save SHAP explainer 
    print("\nStep 6: Building and saving SHAP explainer...")
    X_train_sample = df[NUMERIC_COLS + BINARY_COLS + OHE_COLS + TARGET_ENCODE_COLS]
    X_transformed = pipeline.named_steps['preprocessor'].transform(X_train_sample)
    explainer = shap.TreeExplainer(pipeline.named_steps['model'], X_transformed)
    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"SHAP explainer saved to {EXPLAINER_PATH}")
    
    # Final message
    print("\nTraining complete! All model artifacts have been saved to the 'models/' directory.")
    