""" 
Temporal features capture how active and established a listing is.
A listing with a recent review is likely more active than one where
the last review was 2 years ago. Listing age is a proxy for how 
established and trusted a listing is on the platform.
"""

import pandas as pd

from src.features.constants import SCRAPE_DATE

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Add time-based features derived from listing and review dates.
    
    :param df: DataFrame containing `last_review`, `first_review`, and `host_since` columns.
    :type df: pd.DataFrame
    
    :return: DataFrame with temporal feature columns added.
    :rtype: pd.DataFrame
    """
    # Days since the listing last received a review (proxy for recency/activity)
    # Null for listings with no reviews yet (imputed with median later)
    df['days_since_last_review'] = (SCRAPE_DATE - pd.to_datetime(df['last_review'])).dt.days
    
    # How long the listing has been active on Airbnb
    df['listing_age_days'] = (SCRAPE_DATE - pd.to_datetime(df['first_review'])).dt.days
    
    return df