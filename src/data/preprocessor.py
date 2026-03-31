""" 
Main preprocessing pipeline, which applies all feature engineering steps to the cleaned DataFrame.
Calls each feature module in sequence and saves the result to parquet
for use in model training and the Streamlit app.
"""

import pandas as pd

from src.data.loader import load_and_preprocess_data
from src.features.amenities import parse_amenities
from src.features.geo import add_geo_features
from src.features.text import add_text_features
from src.features.bathrooms import parse_bathrooms
from src.features.host import add_host_features
from src.features.temporal import add_temporal_features
from src.features.bedroom import add_bedroom_features

def build_features(filepath: str) -> pd.DataFrame:
    """ 
    Load the raw data and apply all feature engineering steps in sequence.

    :param filepath: Path to the raw CSV file containing Airbnb listings data.
    :type filepath: str

    :returns: A DataFrame with all features engineered and ready for modeling.
    :rtype: pd.DataFrame
    """
    df = load_and_preprocess_data(filepath)
    
    df = parse_amenities(df)
    df = add_geo_features(df)
    df = add_text_features(df)
    df = parse_bathrooms(df)
    df = add_host_features(df)
    df = add_temporal_features(df)
    df = add_bedroom_features(df)

    return df

def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """ 
    Save the processed DataFrame to a parquet file for efficient storage and loading.

    :param df: The processed DataFrame containing engineered features.
    :type df: pd.DataFrame
    :param output_path: The file path where the processed data should be saved.
    :type output_path: str
    """
    df.to_parquet(output_path, index=False)