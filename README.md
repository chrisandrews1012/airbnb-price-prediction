# Edinburgh Airbnb Price Predictor

![GitHub last commit](https://img.shields.io/github/last-commit/chrisandrews1012/airbnb-price-prediction)
![GitHub repo size](https://img.shields.io/github/repo-size/chrisandrews1012/airbnb-price-prediction)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Stack](https://img.shields.io/badge/stack-LightGBM%20%7C%20Streamlit%20%7C%20SHAP%20%7C%20Optuna-blue)

Predicts nightly Airbnb prices for Edinburgh listings and explains what drives them.

## Problem Statement

Edinburgh listings vary wildly by location, amenity set, host tenure, and listing quality. A raw price estimate isn't that useful on its own. The challenge is building a model that captures enough signal across all those dimensions while staying interpretable enough that a host can actually act on the output.

## Approach

Feature engineering runs across six modules: geospatial distances to key Edinburgh landmarks, amenity parsing with binary flags for high-signal items, host tenure and response metrics, temporal signals from review history, text features from listing titles and descriptions, and bathroom parsing from free text.

Four models were benchmarked on the same train/test split: Ridge, Random Forest, XGBoost, and LightGBM. LightGBM won on all metrics and was tuned with Optuna (100 trials, 5-fold cross-validation). A SHAP TreeExplainer is fitted on the training set and used at inference time to show which features drove each prediction.

The app is a four-page Streamlit interface covering price prediction, market analysis, neighbourhood exploration, and model insights.

## Results

Evaluated on a held-out test set of 967 Edinburgh listings (September 2025 snapshot, 4,832 listings after cleaning).

| Metric | Value |
|---|---|
| R² | 0.666 |
| RMSE | £78.58 |
| MAE | £43.74 |
| Median AE | £23.44 |
| MAPE | 22.2% |

For more than half of all predictions the model is within £23.44 of the actual price.

## How to Run

```bash
git clone https://github.com/chrisandrews1012/airbnb-price-prediction.git
cd airbnb-price-prediction
uv sync
```

The trained model artifacts are included in the repo, so you can run the app straight away:

```bash
uv run streamlit run app.py
```

To retrain from scratch, download `listings.csv` for Edinburgh from [Inside Airbnb](http://insideairbnb.com/get-the-data/), place it at `data/raw/listings.csv`, and run:

```bash
uv run python scripts/train_model.py
```

**Docker**

```bash
docker compose up --build
```

## File Structure

```
airbnb-price-prediction/
├── app.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── scripts/
│   └── train_model.py
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
├── reports/
│   └── figures/
└── src/
    └── airbnb_price_prediction/
        ├── app/
        ├── data/
        ├── features/
        └── models/
```

## License

This project is licensed under the [MIT License](LICENSE).
