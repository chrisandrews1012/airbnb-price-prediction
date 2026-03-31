""" 
Model training pipeline.
Defines feature groups, builds the sklearn pipeline, and runs Optuna to findthe best LightGBM hyperparameters.
"""

import joblib
import json
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# Explicitly defining column groups for clarity and to avoid hardcoding column names in the pipeline
# This makes it easy to add/remove features without having to change the pipeline code

NUMERIC_COLS = [
    "accommodates", "bedrooms", "beds", "bathrooms_parsed",
    "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "review_scores_rating", "review_scores_cleanliness", "review_scores_location",
    "review_scores_value", "review_scores_accuracy", "review_scores_checkin",
    "review_scores_communication", "calculated_host_listings_count",
    "host_acceptance_rate", "host_response_rate", "host_tenure_days",
    "days_since_last_review", "listing_age_days", "beds_per_bedroom",
    "amenity_count", "name_length", "description_length",
    "dist_to_castle_km", "dist_to_station_km", "dist_to_royal_mile_km",
    "dist_to_airport_km", "estimated_occupancy_l365d",
]

BINARY_COLS = [
    "host_is_superhost", "host_identity_verified", "host_has_profile_pic",
    "instant_bookable", "is_shared_bath", "is_old_town", "is_new_town",
    "has_premium_keyword", "has_wifi", "has_kitchen", "has_washer", "has_dryer",
    "has_parking", "has_pool", "has_hot_tub", "has_gym", "has_ev_charger",
    "has_air_conditioning", "has_dishwasher", "has_dedicated_workspace",
    "has_long_term_stays_allowed",
]

# One-hot encode low-cardinality categorical columns (e.g., room type, property type)
OHE_COLS = ["room_type", "property_type"]

# Target encoding for high-cardinality categorical columns (e.g., neighborhood)
""" 
Target encode neighbourhood_cleansed, which replaces each neighbourhood with the average log_price 
of listings in that neighbourhood (e.g., "Old Town" -> 5.4, "Leith" -> 4.9).

This captures the price signal of each neighborhood as a single number, which is more 
informative than one-hot encoding 30 neighbourhoods into 30 sparse binary columns. 

Note: sklearn's TargetEncoder  uses cross-fitting internally to prevent data leakage. 
"""
TARGET_ENCODE_COLS = ["neighbourhood_cleansed"]


def build_pipeline() -> Pipeline:
    """
    Build the sklearn preprocessing and modelling pipeline.

    Numeric and binary columns are imputed with the median and scaled.
    Categorical columns are one-hot encoded or target encoded.

    :returns: Unfitted sklearn Pipeline.
    :rtype: Pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_COLS + BINARY_COLS,
            ),
            (
                "ohe",
                # handle_unknown="ignore" prevents crashes if the app sends an unseen category
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                OHE_COLS,
            ),
            (
                "target_enc",
                # smooth="auto" lets sklearn decide regularisation strength based on sample size
                TargetEncoder(smooth="auto"),
                TARGET_ENCODE_COLS,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        # verbose=-1 silences LightGBM training output
        ("model", LGBMRegressor(random_state=42, verbose=-1)),
    ])