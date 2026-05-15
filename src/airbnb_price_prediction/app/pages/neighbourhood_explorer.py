import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/listings_clean.parquet")


df = load_data()

st.markdown("## Neighbourhood Explorer")
st.markdown(
    '<p class="page-subtitle">Explore pricing and listing statistics by Edinburgh neighbourhood.</p>',
    unsafe_allow_html=True,
)
st.divider()

neighbourhood = st.selectbox(
    "Select a neighbourhood",
    sorted(df["neighbourhood_cleansed"].unique()),
)

neighbourhood_df = df[df["neighbourhood_cleansed"] == neighbourhood]

# Key Stats
st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

median_price = neighbourhood_df["price"].median()
listing_count = len(neighbourhood_df)
avg_rating = neighbourhood_df["review_scores_rating"].mean()
superhost_pct = neighbourhood_df["host_is_superhost"].mean() * 100

st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-value">£{median_price:.0f}</div>
            <div class="stat-label">Median Price/Night</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{listing_count:,}</div>
            <div class="stat-label">Listings</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{avg_rating:.2f}</div>
            <div class="stat-label">Avg Rating</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{superhost_pct:.0f}%</div>
            <div class="stat-label">Superhosts</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# Room Type Breakdown 
st.markdown('<div class="section-header">Price by Room Type</div>', unsafe_allow_html=True)
st.caption("Median nightly price per room type in this neighbourhood.")

room_type_df = (
    neighbourhood_df
    .groupby("room_type")
    .agg(median_price=("price", "median"), listing_count=("price", "count"))
    .reset_index()
    .sort_values("median_price", ascending=False)
)

for _, row in room_type_df.iterrows():
    st.markdown(
        f"""
        <div class="listing-card">
            <div>
                <div class="listing-card-title">{row['room_type']}</div>
                <div class="listing-card-subtitle">{int(row['listing_count'])} listings</div>
            </div>
            <div>
                <div class="listing-card-price">£{row['median_price']:.0f}</div>
                <div class="listing-card-subtitle" style="text-align:right">median/night</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Capacity Breakdown 
st.markdown('<div class="section-header">Price by Capacity</div>', unsafe_allow_html=True)
st.caption("Median nightly price by number of guests accommodated.")

capacity_df = (
    neighbourhood_df
    .groupby("accommodates")
    .agg(median_price=("price", "median"), listing_count=("price", "count"))
    .reset_index()
    .sort_values("accommodates")
)

for _, row in capacity_df.iterrows():
    guests = int(row["accommodates"])
    guest_label = f"{guests} guest" if guests == 1 else f"{guests} guests"
    st.markdown(
        f"""
        <div class="listing-card">
            <div>
                <div class="listing-card-title">{guest_label}</div>
                <div class="listing-card-subtitle">{int(row['listing_count'])} listings</div>
            </div>
            <div>
                <div class="listing-card-price">£{row['median_price']:.0f}</div>
                <div class="listing-card-subtitle" style="text-align:right">median/night</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
