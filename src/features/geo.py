"""
Location is one of the strongest price signals in the Airbnb dataset. 
Raw latitude and longitude coordinates are not meaningful to a model on their own,
so we need to convert coordinates into distances to key Edinburgh landmarks.

E.g., a listing 200m from the Edinburgh Castle commands a very different price 
to one 5km away, even if all other features are identical.

Will use the haversine formula to calculate distances because the Earth is curved.
Simple coordinate differences would be inaccurate, especially for listings farther from the landmarks.
The formula is implemented using numpy so it runs across the entire latitude/longitude column at once
(vectorised) rather than looping row by row, so no external geospatial libraries are needed.
"""
import pandas as pd
import numpy as np

# Edinburgh landmark coordinates (latitude, longitude)
LANDMARKS = {
    "dist_to_castle_km":(55.9486, -3.1999),       # Edinburgh Castle
    "dist_to_royal_mile_km": (55.9503, -3.1883),  # Royal Mile
    "dist_to_station_km": (55.9520, -3.1884),     # Waverley Station
    "dist_to_airport_km": (55.9500, -3.3725),     # Edinburgh Airport
}

OLD_TOWN_NEIGHBORHOODS = {"Old Town", "Canongate", "Grassmarket"}
NEW_TOWN_NEIGHBORHOODS = {"New Town", "Stockbridge"}

def _haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """
    Calculate haversine distance in kilometers between a column of coordinates and a fixed point.
    
    :param lat1: Array of latitudes for the listings (degrees).
    :type lat1: np.ndarray
    :param lon1: Array of longitudes for the listings (degrees).
    :type lon1: np.ndarray
    :param lat2: Latitude of the fixed point (degrees).
    :type lat2: float
    :param lon2: Longitude of the fixed point (degrees).
    :type lon2: float
    
    :return: Array of distances in kilometers.
    :rtype: np.ndarray
    """
    R = 6371  # Radius of Earth in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geospatial distance and neighborhood flag features. 
    
    :param df: DataFrame containing 'latitude', 'longitude', and 'neighbourhood_cleansed' columns.
    :type df: pd.DataFrame
    
    :return: DataFrame with distance columns and neighborhood binary flags added.
    :rtype: pd.DataFrame
    """
    for col, (lat, lon) in LANDMARKS.items():
        df[col] = _haversine(df['latitude'].values, df['longitude'].values, lat, lon)
        
    # Binary neighborhood flags for Old Town and New Town, which are popular and pricier areas
    df['is_old_town'] = df['neighbourhood_cleansed'].isin(OLD_TOWN_NEIGHBORHOODS).astype(int)
    df['is_new_town'] = df['neighbourhood_cleansed'].isin(NEW_TOWN_NEIGHBORHOODS).astype(int)
    
    return df