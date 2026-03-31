"""Data loading and preprocessing function for Airbnb listings data."""

import pandas as pd
import numpy as np

# Columns with no practical use for modeling (e.g. URLs, IDs, timestamps)
COLS_TO_DROP = [
    "id",
    "listing_url",
    "scrape_id",
    "last_scraped",
    "source",
    "neighborhood_overview",         # 2160 nulls, free text
    "picture_url",
    "host_id",
    "host_url",
    "host_name",                     # PII, no predictive value
    "host_location",                 # 950 nulls, redundant
    "host_about",                    # free text, high null rate
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",            # 3441/4936 null
    "host_total_listings_count",     # redundant with calculated_host_listings_count
    "host_verifications",            # low-signal list string
    "neighbourhood",                 # redundant with neighbourhood_cleansed
    "neighbourhood_group_cleansed",  # 100% null
    "calendar_updated",              # 100% null
    "license",                       # 75% null
    "bathrooms"                      # almost entirely null
]

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Read the raw InsideAirbnb data and preprocess it for modeling.
    
    :param filepath: Path to the raw CSV file containing Airbnb listings data.
    :type filepath: str
    
    :return: A cleaned and preprocessed DataFrame ready for analysis and modeling.
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Drop irrelevant columns identified during EDA
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    
    # Parse price column: remove $ and commas, convert to float
    df['price'] = df['price'].str.replace('[\$,]', '', regex=True).astype(float)
    
    # Remove null, zero, and extreme outlier prices (above 99th percentile)
    p99 = df['price'].quantile(0.99)
    df = df[(df['price'].notna() & df['price'] > 0) & (df['price'] <= p99)]
    
    # Log-transform the price to reduce skewness - this is what the model will train on
    # Price is right-skewed (a few very expensive listings), so log transformation helps normalize it
    # log1p compresses the scale so the model treats proportional errors equally across the price range
    # instead of over-fitting to the expensive listings
    # log1p(x) = log(1 + x). Predictions are converted back with np.expm1 before being displayed to users
    df['log_price'] = np.log1p(df['price'])
    
    df = df.reset_index(drop=True)
    
    return df