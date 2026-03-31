""" 
Inference interface for the trained model.

Used by the API to generate price predictions and SHAP (SHapley Additive exPlanations) values 
for incoming user requests.
"""

import joblib
import pandas as pd
import numpy as np 
import shap
from sklearn.pipeline import Pipeline

def load_pipeline(path: str = "models/pipeline.joblib") -> Pipeline:
    """
    Load the trained sklearn pipeline from disk.

    :param path: Path to the saved pipeline.
    :type path: str

    :returns: Fitted sklearn Pipeline.
    :rtype: Pipeline
    """
    return joblib.load(path)

def load_explainer(path: str = "models/shap_explainer.joblib") -> shap.TreeExplainer:
    """
    Load the cached SHAP TreeExplainer from disk.

    :param path: Path to the saved explainer.
    :type path: str

    :returns: SHAP TreeExplainer.
    :rtype: shap.TreeExplainer
    """
    return joblib.load(path)

def predict_price(pipeline: Pipeline, input_df: pd.DataFrame) -> float:
    """
    Predict the nightly price in £ for a single listing.

    :param pipeline: Fitted sklearn Pipeline.
    :type pipeline: Pipeline
    :param input_df: Single-row DataFrame with all required feature columns.
    :type input_df: pd.DataFrame

    :returns: Predicted nightly price in £.
    :rtype: float
    """
    log_price = pipeline.predict(input_df)[0]  # Get the log-price prediction
    # Convert back from log-price to original price scale (£)
    return float(np.expm1(log_price))

def get_shap_values(pipeline: Pipeline, explainer: shap.TreeExplainer, input_df: pd.DataFrame) -> tuple:
    """
    Compute SHAP values for a single prediction to explain which features
    drove the price up or down.

    :param pipeline: Fitted sklearn Pipeline.
    :type pipeline: Pipeline
    :param explainer: SHAP TreeExplainer fitted on the training set.
    :type explainer: shap.TreeExplainer
    :param input_df: Single-row DataFrame with all required feature columns.
    :type input_df: pd.DataFrame

    :returns: Tuple of (shap_values array, feature_names list).
    :rtype: tuple
    """
    # Transform input through the preprocessor before passing to SHAP
    X_transformed = pipeline.named_steps['preprocessor'].transform(input_df)
    shap_values = explainer.shap_values(X_transformed)
    
    # Get feature names from the preprocessor for interpretability
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    return shap_values, feature_names