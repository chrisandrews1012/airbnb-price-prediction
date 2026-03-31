""" 
Listing titles often signal how a host is positioning their property.
Premium keywords like "luxury" or "stunning" suggest higher-end listings
and tend to correlate with higher prices.

Name and description length are a proxy for listing quality and host effort.
E.g., more professional hosts tend to write more detailed listing descriptions,
and these listings tend to command higher prices.
"""

import pandas as pd

# Keywords hosts use to signal a premium listing. These are based on EDA of the most expensive listings.
PREMIUM_KEYWORDS = [
    "luxury", "stunning", "charming", "modern",
    "spacious", "cozy", "boutique", "elegant", "stylish",
]

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Add text-derived features from the name and description columns.
    
    :param df: DataFrame containing 'name' and 'description' columns.
    :type df: pd.DataFrame

    :return: DataFrame with new text features added.
    :rtype: pd.DataFrame
    """
    name = df['name'].fillna('')
    description = df['description'].fillna('')
    
    # Length of the listing title and description 
    df['name_length'] = name.str.len()
    df['description_length'] = description.str.len()
    
    # 1 if the listing name contains any premium keywords, else 0
    pattern = "|".join(PREMIUM_KEYWORDS)
    df['has_premium_keyword'] = name.str.contains(pattern, case=False, regex=True).astype(int)
    
    return df