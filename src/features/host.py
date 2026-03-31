""" 
Host-related features capture the quality and experience of the person
running the listing. More experienced, responsive, and verified hosts 
tend to charge more and receive better reviews.
"""

import pandas as pd

# The date the Inside Airbnb data was scraped.
# Used to calculate temporal features like host_tenure_days and days_since_review.
# Update this if using a newer or older scrape of the data.  
SCRAPE_DATE = pd.Timestamp("2025-09-21")

# Inside Airbnb stores booleans as "t" and "f" strings, not Python True/False
BOOL_COLS = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]

# Ordinal encoding (e.g., faster response times = higher score)
HOST_RESPONSE_TIME_MAP = {
    "within an hour": 4,
    "within a few hours": 3,
    "within a day": 2,
    "a few days or more": 1,
}

def add_host_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Engineer host-related features from raw host columns.

    :param df: DataFrame containing raw host columns.
    :type df: pd.DataFrame

    :returns: DataFrame with host feature columns added.
    :rtype: pd.DataFrame
    """
    # Parse percentage strings into floats (e.g., "95%" -> 95.0)
    for col in ["host_response_rate", "host_acceptance_rate"]:
        df[col] = df[col].str.replace("%", "", regex=False).astype(float)
    
    # Convert "t"/"f" strings to 1/0 integers
    for col in BOOL_COLS:
        df[col] = df[col].map({"t": 1, "f": 0})
        
    # Ordinal encoding of host response time (null means no response time listed)
    df['host_response_time'] = (
        df['host_response_time']
        .map(HOST_RESPONSE_TIME_MAP)
        .fillna(0)
        .astype(int)
    )
    
    # Number of days the host has been on Airbnb (proxy for experience)
    df['host_tenure_days'] = (SCRAPE_DATE - pd.to_datetime(df['host_since'])).dt.days
    
    return df