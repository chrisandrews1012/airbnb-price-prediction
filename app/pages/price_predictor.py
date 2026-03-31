import joblib
import pandas as pd
import streamlit as st
from src.models.predict import predict_price

@st.cache_resource
def load_pipeline():
    return joblib.load("models/pipeline.joblib")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/listings_clean.parquet")

pipeline = load_pipeline()
df = load_data()

st.markdown("## Price Predictor")
st.markdown('<p class="page-subtitle">Enter your listing details to get an estimated nightly price based on Edinburgh market data.</p>', unsafe_allow_html=True)
st.divider()

col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:

    st.markdown('<div class="section-header">Property</div>', unsafe_allow_html=True)

    room_type = st.selectbox(
        "Room Type",
        df["room_type"].unique(),
        index=list(df["room_type"].unique()).index("Entire home/apt"),
    )
    property_type = st.selectbox(
        "Property Type",
        sorted(df["property_type"].unique()),
        index=sorted(df["property_type"].unique()).index("Entire rental unit"),
    )
    neighbourhood = st.selectbox(
        "Neighbourhood",
        sorted(df["neighbourhood_cleansed"].unique()),
        index=sorted(df["neighbourhood_cleansed"].unique()).index(
            "Old Town, Princes Street and Leith Street"
        ),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        accommodates = st.slider("Accommodates", 1, 16, 3)
        bedrooms = st.slider("Bedrooms", 0, 10, 1)
    with col_b:
        beds = st.slider("Beds", 1, 16, 2)
        bathrooms = st.slider("Bathrooms", 0.5, 10.0, 1.0, step=0.5)

    st.markdown('<div class="section-header">Amenities</div>', unsafe_allow_html=True)

    amenities = st.multiselect(
        "Select amenities your listing has",
        [
            "WiFi", "Kitchen", "Washer", "Dryer", "Parking",
            "Pool", "Hot Tub", "Gym", "EV Charger",
            "Air Conditioning", "Dishwasher", "Dedicated Workspace",
            "Long Term Stays Allowed",
        ],
        default=["WiFi", "Kitchen"],
    )
    st.caption("Showing the 13 amenities found to meaningfully influence nightly price. Listings may advertise many more.")

    st.markdown('<div class="section-header">Host</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)
    with col_c:
        is_superhost = st.toggle("Superhost", value=False)
        instant_bookable = st.toggle("Instant Bookable", value=False)
    with col_d:
        host_response_time = st.selectbox(
            "Response Time",
            ["within an hour", "within a few hours", "within a day", "a few days or more"],
        )
        host_tenure_years = st.slider("Years on Airbnb", 0, 15, 9)

    st.markdown("")
    predict_btn = st.button("Predict Price")

with col_results:
    if predict_btn:
        amenity_map = {
            "WiFi": "has_wifi", "Kitchen": "has_kitchen", "Washer": "has_washer",
            "Dryer": "has_dryer", "Parking": "has_parking", "Pool": "has_pool",
            "Hot Tub": "has_hot_tub", "Gym": "has_gym", "EV Charger": "has_ev_charger",
            "Air Conditioning": "has_air_conditioning", "Dishwasher": "has_dishwasher",
            "Dedicated Workspace": "has_dedicated_workspace",
            "Long Term Stays Allowed": "has_long_term_stays_allowed",
        }

        response_time_map = {
            "within an hour": 4, "within a few hours": 3,
            "within a day": 2, "a few days or more": 1,
        }

        neighbourhood_data = df[df["neighbourhood_cleansed"] == neighbourhood].iloc[0]

        input_data = {
            "room_type": room_type,
            "property_type": property_type,
            "neighbourhood_cleansed": neighbourhood,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "beds": beds,
            "bathrooms_parsed": bathrooms,
            "is_shared_bath": 0,
            "amenity_count": len(amenities),
            "host_is_superhost": int(is_superhost),
            "host_has_profile_pic": 1,
            "host_identity_verified": 1,
            "instant_bookable": int(instant_bookable),
            "host_response_time": response_time_map[host_response_time],
            "host_response_rate": 95.0,
            "host_acceptance_rate": 90.0,
            "host_tenure_days": host_tenure_years * 365,
            "beds_per_bedroom": beds / max(bedrooms, 1),
            "name_length": 50,
            "description_length": 500,
            "has_premium_keyword": 0,
            "minimum_nights": 2,
            "maximum_nights": 365,
            "availability_30": 15,
            "availability_60": 30,
            "availability_90": 45,
            "availability_365": 180,
            "number_of_reviews": 10,
            "number_of_reviews_ltm": 5,
            "reviews_per_month": 1.0,
            "review_scores_rating": 4.5,
            "review_scores_cleanliness": 4.5,
            "review_scores_location": 4.5,
            "review_scores_value": 4.5,
            "review_scores_accuracy": 4.5,
            "review_scores_checkin": 4.5,
            "review_scores_communication": 4.5,
            "calculated_host_listings_count": 1,
            "estimated_occupancy_l365d": 0.55,
            "days_since_last_review": 30,
            "listing_age_days": 365,
            "dist_to_castle_km": neighbourhood_data["dist_to_castle_km"],
            "dist_to_station_km": neighbourhood_data["dist_to_station_km"],
            "dist_to_royal_mile_km": neighbourhood_data["dist_to_royal_mile_km"],
            "dist_to_airport_km": neighbourhood_data["dist_to_airport_km"],
            "is_old_town": neighbourhood_data["is_old_town"],
            "is_new_town": neighbourhood_data["is_new_town"],
        }

        for label, col in amenity_map.items():
            input_data[col] = 1 if label in amenities else 0

        input_df = pd.DataFrame([input_data])
        price = predict_price(pipeline, input_df)

        # Confidence range using model MAE (£43.74)
        mae = 43.74
        low = max(0, price - mae)
        high = price + mae

        st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Nightly Price</div>
                <div class="prediction-price">£{price:.0f}</div>
                <div class="prediction-label">per night</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="stat-row">
                <div class="stat-item">
                    <div class="stat-value">£{low:.0f}</div>
                    <div class="stat-label">Lower Estimate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">£{price:.0f}</div>
                    <div class="stat-label">Predicted</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">£{high:.0f}</div>
                    <div class="stat-label">Upper Estimate</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Similar listings
        similar = df[
            (df["room_type"] == room_type) &
            (df["neighbourhood_cleansed"] == neighbourhood)
        ][["price", "accommodates", "bedrooms", "amenity_count", "review_scores_rating"]].head(5)

        if len(similar) > 0:
            st.markdown('<div class="section-header">Similar Listings</div>', unsafe_allow_html=True)
            st.caption(f"Real listings from the Edinburgh dataset matching your selected room type ({room_type}) and neighbourhood.")
            for _, row in similar.iterrows():
                rating = row["review_scores_rating"]
                rating_str = f"{rating:.1f} / 5" if pd.notna(rating) else "No reviews"
                st.markdown(f"""
                    <div class="listing-card">
                        <div>
                            <div class="listing-card-title">
                                {int(row['accommodates'])} guests · {int(row['bedrooms'])} bed · {int(row['amenity_count'])} total amenities
                            </div>
                            <div class="listing-card-subtitle">Rating: {rating_str}</div>
                        </div>
                        <div class="listing-card-price">£{row['price']:.0f}</div>
                    </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <div class="info-box">
                Fill in your listing details and click <strong>Predict Price</strong>
                to get your estimated nightly rate and comparable listings.
            </div>
        """, unsafe_allow_html=True)
