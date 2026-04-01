# Edinburgh Airbnb Host Intelligence Platform

A full-stack machine learning application that predicts nightly listing prices for Edinburgh Airbnb hosts. Built with LightGBM, Streamlit, containerized with Docker.

## Overview

The project follows an end-to-end ML pipeline, from raw data through to a deployed Streamlit application.

**1. Data Ingestion**:
Raw listing data is loaded from the Inside Airbnb Edinburgh dataset (September 2025 snapshot, 4,832 listings after cleaning). Prices are parsed, capped at £900 to remove extreme outliers, and log-transformed to normalize the distribution before modeling.

**2. Feature Engineering**:
Features are extracted across several modules:
- **Geospatial**: Haversine distances to Edinburgh Castle, Waverley Station, the Royal Mile, and the airport. Binary flags for Old Town and New Town.
- **Amenities**: Total amenity count plus 13 binary flags for amenities found to influence price (WiFi, pool, hot tub, EV charger, etc.)
- **Host**: Tenure in days, response time (ordinal encoded), response rate, superhost and instant bookable flags.
- **Temporal**: Days since last review, listing age in days.
- **Text**: Title and description length, premium keyword flag.
- **Bathrooms**: Parsed from free-text, shared bath flag, half-bath handling.

**3. Modeling**:
Four models were benchmarked on the same train/test split: Ridge, Random Forest, XGBoost, and LightGBM. LightGBM won on all metrics and was selected for hyperparameter tuning with Optuna (100 trials, 5-fold cross-validation). The final model explains 66.6% of price variance with a median absolute error of £23.44.

**4. Explainability** *(work in progress)*:
A SHAP TreeExplainer is fitted on the training set and saved alongside the pipeline. At inference time, SHAP values are computed per prediction and filtered to only the features the user can control, giving a per-input breakdown of what drove the predicted price.

**5. Application**:
A four-page Streamlit app provides price prediction, market analysis, neighbourhood exploration, and model insights. All pages share a consistent dark-theme CSS. The app is containerised with Docker for reproducible deployment.

---

## Features

- **Price Predictor**: Enter your listing details and get a predicted nightly price with a confidence range and SHAP-based breakdown of the factors driving it *(SHAP breakdown is a work in progress)*
- **Market Analysis**: Explore pricing across room types, neighbourhoods, and guest capacity
- **Neighbourhood Explorer**: Drill into per-neighbourhood pricing and listing statistics
- **Model Insights**: Model performance metrics, benchmark comparisons, feature importances, and notes on possible improvements

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, numpy |
| Feature engineering | scikit-learn, haversine |
| Modeling | LightGBM, XGBoost, Random Forest, Ridge |
| Hyperparameter tuning | Optuna (100 trials, 5-fold CV) |
| Explainability | SHAP TreeExplainer |
| Frontend | Streamlit |
| Containerization | Docker |

---

## Model Performance

Evaluated on a held-out test set of 967 Edinburgh listings (September 2025).

| Metric | Value |
|---|---|
| RMSE | £78.58 |
| MAE | £43.74 |
| MAPE | 22.2% |
| R² | 0.666 |
| Median AE | £23.44 |

> For more than half of all predictions the model is within £23.44 of the actual price.

---

## Project Structure

```
├── app/
│   ├── pages/
│   │   ├── price_predictor.py
│   │   ├── market_analysis.py
│   │   ├── neighbourhood_explorer.py
│   │   └── model_insights.py
│   └── style.css
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/
│   │   ├── amenities.py
│   │   ├── bathrooms.py
│   │   ├── bedroom.py
│   │   ├── constants.py
│   │   ├── geo.py
│   │   ├── host.py
│   │   ├── temporal.py
│   │   └── text.py
│   └── models/
│       ├── benchmark.py
│       ├── evaluate.py
│       ├── predict.py
│       └── train.py
├── scripts/
│   └── train_model.py
├── models/
│   ├── pipeline.joblib
│   ├── shap_explainer.joblib
│   ├── benchmark_results.json
│   └── model_metadata.json
├── data/
│   ├── listings.csv
│   └── processed/
│       └── listings_clean.parquet
├── app.py
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## How to Run

### With Docker (recommended)

```bash
docker compose up --build
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Without Docker

Requires [uv](https://github.com/astral-sh/uv).

The raw dataset is not included in this repository due to its size. Before running the app you will need to:

1. Download `listings.csv` for Edinburgh from [Inside Airbnb](http://insideairbnb.com/get-the-data/) and place it at `data/listings.csv`
2. Run the training pipeline to generate the processed data and model files:
```bash
uv run python scripts/train_model.py
```
3. Then start the app:
```bash
uv sync
uv run streamlit run app.py
```

---

## Data

This project uses the Edinburgh listings dataset from [Inside Airbnb](http://insideairbnb.com/get-the-data/) (September 2025 snapshot).

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: Inside Airbnb.

The raw `listings.csv` is not included in this repository. To retrain the model from scratch:

1. Download `listings.csv` for Edinburgh from Inside Airbnb
2. Place it at `data/listings.csv`
3. Run `uv run python scripts/train_model.py`
