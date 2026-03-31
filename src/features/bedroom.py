""" 
Bedroom-derived features capture the relationship between beds and bedrooms.
A listing with 4 beds in 1 bedroom is very different to one with 1 bed per bedroom,
even if the total bed count is the same.
"""

import pandas as pd

def add_bedroom_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Engineer bedroom-related features from raw bed and bedroom columns.

    :param df: DataFrame containing raw `beds` and `bedrooms` columns.
    :type df: pd.DataFrame

    :returns: DataFrame with bedroom feature columns added.
    :rtype: pd.DataFrame
    """
    # Beds per bedroom (proxy for how spacious or densely packed a listing is)
    df['beds_per_bedroom'] = df['beds'] / df['bedrooms'].replace(0, 1)
    
    return df