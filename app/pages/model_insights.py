import json
import joblib
import pandas as pd
import streamlit as st


@st.cache_data
def load_metadata():
    with open("models/model_metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_benchmark():
    with open("models/benchmark_results.json") as f:
        return json.load(f)


@st.cache_resource
def load_pipeline():
    return joblib.load("models/pipeline.joblib")


metadata = load_metadata()
benchmark = load_benchmark()
pipeline = load_pipeline()

st.markdown("## Model Insights")
st.markdown(
    '<p class="page-subtitle">Performance metrics, model comparisons, and feature importances for the trained LightGBM model.</p>',
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Performance", "Benchmark", "Feature Importance", "What I'd Do Next Time"]
)

# Performance 
with tab1:
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    st.caption("Evaluated on a held-out test set of 967 listings. All metrics are in £ after converting back from log price.")

    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-value">£{metadata['rmse']:.2f}</div>
                <div class="stat-label">RMSE</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">£{metadata['mae']:.2f}</div>
                <div class="stat-label">MAE</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{metadata['mape']:.1f}%</div>
                <div class="stat-label">MAPE</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{metadata['r2']:.3f}</div>
                <div class="stat-label">R²</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">£{metadata['median_ae']:.2f}</div>
                <div class="stat-label">Median AE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    metrics_info = [
        ("RMSE: £{:.2f}".format(metadata['rmse']),
         "Average prediction error in £, with larger errors penalised more heavily. Pulled up by a small number of hard-to-predict listings."),
        ("MAE: £{:.2f}".format(metadata['mae']),
         "Average prediction error in £, treating all errors equally. More representative than RMSE for typical performance."),
        ("MAPE: {:.1f}%".format(metadata['mape']),
         "Average error as a percentage of the actual price. Scale-independent — a £20 error on a £50 listing is weighted more than on a £500 listing."),
        ("R²: {:.3f}".format(metadata['r2']),
         "The model explains {:.1f}% of the variation in nightly price across Edinburgh listings.".format(metadata['r2'] * 100)),
        ("Median AE: £{:.2f}".format(metadata['median_ae']),
         "For more than half of all listings, the model's prediction is within £{:.2f} of the actual price. The most honest measure of typical performance.".format(metadata['median_ae'])),
    ]

    for title, description in metrics_info:
        st.markdown(
            f"""
            <div class="listing-card">
                <div>
                    <div class="listing-card-title">{title}</div>
                    <div class="listing-card-subtitle">{description}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Benchmark 
with tab2:
    st.markdown('<div class="section-header">Model Benchmark</div>', unsafe_allow_html=True)
    st.caption("All four models evaluated on the same train/test split. Metrics are in log price units.")

    # Sort by RMSE ascending so best model appears first
    benchmark_sorted = sorted(benchmark, key=lambda x: x["rmse"])

    for i, result in enumerate(benchmark_sorted):
        rank_label = "Winner" if i == 0 else f"#{i + 1}"
        st.markdown(
            f"""
            <div class="listing-card">
                <div>
                    <div class="listing-card-title">{result['model']} <span style="color:#a0a0b0; font-weight:400; font-size:0.85rem">— {rank_label}</span></div>
                    <div class="listing-card-subtitle">RMSE {result['rmse']:.4f} · MAE {result['mae']:.4f} · R² {result['r2']:.4f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Feature importance 
with tab3:
    st.markdown('<div class="section-header">Top Feature Importances</div>', unsafe_allow_html=True)
    st.caption("Features ranked by how much they reduce prediction error across all trees in the LightGBM model.")

    st.markdown("""
        <div class="listing-card" style="display:block">
            <div class="listing-card-title">How These are Calculated:</div><br>
            <div class="listing-card-subtitle">
                LightGBM builds hundreds of decision trees sequentially. At each node in every tree, it picks the feature that reduces prediction error the most and splits the data on it. Feature importance is the total number of times each feature was chosen as a split point across all trees (e.g., features chosen more often score higher).<br><br>
                The raw scores are split counts, not percentages. They are only meaningful relative to each other, which is why they are shown as a normalised bar.<br><br>
                <strong>Known limitation:</strong> Split count tends to overrate high-cardinality features like neighbourhood (many unique values = many split opportunities) and underrate binary features like has_pool (only two values, fewer splits possible).
            </div>
        </div>
    """, unsafe_allow_html=True)

    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_importances = importances.nlargest(15).sort_values(ascending=False)

    max_importance = top_importances.iloc[0]

    for feature, importance in top_importances.items():
        # Strip transformer prefix for display
        display_name = feature.split("__", 1)[-1].replace("_", " ").title()
        bar_width = int((importance / max_importance) * 100)
        st.markdown(
            f"""
            <div class="listing-card">
                <div style="width:100%">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem">
                        <div class="listing-card-title">{display_name}</div>
                        <div class="listing-card-subtitle">{importance:,}</div>
                    </div>
                    <div style="background:#2e2e3e; border-radius:4px; height:6px; width:100%">
                        <div style="background:#ff4b4b; border-radius:4px; height:6px; width:{bar_width}%"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# What I'd Do Next Time
with tab4:
    st.markdown('<div class="section-header">Possible Enhancements</div>', unsafe_allow_html=True)

    sections = [
        (
            "1. More Feature Engineering",
            [
                ("Seasonal Pricing", "The September snapshot misses how prices change across the year, especially around the Fringe festival in August."),
                ("Text Embeddings", "Running the full listing description and title through a sentence embedding model would capture far more signal than just length and keyword flags."),
                ("Host Portfolio Features", "Hosts with multiple listings often price differently. Aggregating host-level pricing behavior could improve predictions."),
                ("Extended Proximity Features", "Proximity to specific venues, restaurants, or transport links beyond the 4 landmarks used could add meaningful signal."),
            ],
        ),
        (
            "2. Better Outlier Handling",
            [
                ("Separate Model for Luxury Listings", "The hard £900 cap was an improvement, but a more principled approach would be to model luxury listings separately. They behave differently enough that one model may not fit both well."),
            ],
        ),
        (
            "3. Temporal Validation",
            [
                ("Time-Based Train/Test Split", "A random split was used, but a time-based split (training on older listings and testing on newer ones) would better simulate real-world deployment."),
            ],
        ),
        (
            "4. More Data",
            [
                ("Multiple Cities", "Only one city's snapshot was used. Training on multiple cities (London, Amsterdam, Barcelona) and fine-tuning on Edinburgh would likely improve generalization significantly."),
            ],
        ),
    ]

    for section_title, items in sections:
        st.markdown(f'<div class="section-header">{section_title}</div>', unsafe_allow_html=True)
        for title, description in items:
            st.markdown(
                f"""
                <div class="listing-card">
                    <div>
                        <div class="listing-card-title">{title}</div>
                        <div class="listing-card-subtitle">{description}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
