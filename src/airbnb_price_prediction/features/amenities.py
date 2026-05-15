import ast
import pandas as pd

# Mapping of amenity names to standardized categories
AMENITY_FLAGS =  {
    "has_wifi": "wifi",
    "has_kitchen": "kitchen",
    "has_washer": "washer",
    "has_dryer": "dryer",
    "has_parking": "parking",
    "has_pool": "pool",
    "has_hot_tub": "hot tub",
    "has_gym": "gym",
    "has_ev_charger": "ev charger",
    "has_air_conditioning": "air conditioning",
    "has_dishwasher": "dishwasher",
    "has_dedicated_workspace": "dedicated workspace",
    "has_long_term_stays_allowed": "long term stays",
}

def _parse_amenities(amenities_str: str) -> list:
    """
    Parse the raw amenities string from the Airbnb dataset into a list of amenity names.
    
    :param amenities_str: The raw amenities string from the dataset (e.g. '["Wifi", "Kitchen", "Washer"]').
    :type amenities_str: str
    
    :return: A list of amenity names.
    :rtype: list
    """
    try:
        return ast.literal_eval(amenities_str)
    except (SyntaxError, ValueError):
        return []
    
def parse_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """    
    Parse the amenities column in the DataFrame and create binary flags for key amenities.
    
    :param df: The input DataFrame containing the raw amenities column.
    :type df: pd.DataFrame
    
    :return: A DataFrame with new binary columns for each key amenity.
    :rtype: pd.DataFrame
    """
    # Convert string to list for each row in the amenities column
    amenity_lists = df['amenities'].fillna('[]').apply(_parse_amenities)
    
    # Total count of amenities can be a useful feature, so we keep it as is
    df['amenity_count'] = amenity_lists.apply(len)
    
    # Create binary flag columns for each amenity of interest
    for col, amenity in AMENITY_FLAGS.items():
        df[col] = amenity_lists.apply(
            lambda lst, kw=amenity: int(any(kw in a.lower() for a in lst))
        )
    
    return df