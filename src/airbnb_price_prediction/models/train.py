""" 
Model training pipeline.
Defines feature groups, builds the sklearn pipeline, and runs Optuna to findthe best LightGBM hyperparameters.
"""

import joblib
import json
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split, cross_val_score

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
                # smooth="auto" lets sklearn decide regularization strength based on sample size
                TargetEncoder(smooth="auto"),
                TARGET_ENCODE_COLS,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(random_state=42, verbose=-1))    # verbose=-1 silences LightGBM training output
    ])
    
def train(df: pd.DataFrame) -> tuple:
    """ 
    Split data, tune hyperparameters with Optuna, and train the final model.
    
    :param df: Fully engineered DataFrame from `build_features()`.
    :type df: pd.DataFrame

    :returns: Tuple of (fitted Pipeline, X_test, y_test)
    :rtype: tuple
    """
    X = df[NUMERIC_COLS + BINARY_COLS + OHE_COLS + TARGET_ENCODE_COLS]
    y = df['log_price']
    
    # Stratify by price quintile to ensure representative splits across price ranges
    price_quintiles = pd.qcut(y, q=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=price_quintiles
    )
    
    def objective(trial): 
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        
        pipeline = build_pipeline()
        pipeline.set_params(model=LGBMRegressor(**params, random_state=42, verbose=-1))
        
        # CV on training set only - test set is strictly for final evaluation after tuning
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, 
            scoring="neg_root_mean_squared_error"
        )
        
        return -scores.mean()
    
    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print("Running Optuna hyperparameter tuning (100 trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    
    print(f"Best CV RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Retrain on full training set using best hyperparameters
    final_pipeline = build_pipeline()
    final_pipeline.set_params(
        model=LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
    )
    final_pipeline.fit(X_train, y_train)
    
    return final_pipeline, X_test, y_test