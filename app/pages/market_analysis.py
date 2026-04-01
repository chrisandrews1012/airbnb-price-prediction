import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/listings_clean.parquet")


df = load_data()

st.markdown("## Market Analysis")
st.markdown(
    '<p class="page-subtitle">Explore the Edinburgh short-term rental market based on September 2025 Inside Airbnb data.</p>',
    unsafe_allow_html=True,
)
st.divider()

room_types = sorted(df["room_type"].unique())
neighbourhoods = sorted(df["neighbourhood_cleansed"].unique())

tab1, tab2, tab3, tab4 = st.tabs(
    ["Price Distribution", "Price by Room Type", "Price by Neighbourhood", "Price vs. Capacity"]
)

# Price Distribution 
with tab1:
    st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)

    selected_room_type = st.selectbox("Room Type", room_types, key="dist_room_type")

    dist_df = df[df["room_type"] == selected_room_type]
    median_price = dist_df["price"].median()
    listing_count = len(dist_df)
    avg_rating = dist_df["review_scores_rating"].mean()

    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-value">£{median_price:.0f}</div>
                <div class="stat-label">Median Nightly Price</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{listing_count:,}</div>
                <div class="stat-label">Listings</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{avg_rating:.2f}</div>
                <div class="stat-label">Avg Rating</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Price by Room Type 
with tab2:
    st.markdown('<div class="section-header">Price by Room Type</div>', unsafe_allow_html=True)

    selected_neighbourhood = st.selectbox("Neighbourhood", neighbourhoods, key="room_type_neighbourhood")

    room_type_df = (
        df[df["neighbourhood_cleansed"] == selected_neighbourhood]
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

# Price by Neighbourhood 
with tab3:
    st.markdown('<div class="section-header">Price by Neighbourhood</div>', unsafe_allow_html=True)

    selected_room_type_2 = st.selectbox("Room Type", room_types, key="neighbourhood_room_type")

    neighbourhood_df = (
        df[df["room_type"] == selected_room_type_2]
        .groupby("neighbourhood_cleansed")
        .agg(median_price=("price", "median"), listing_count=("price", "count"))
        .reset_index()
        .sort_values("median_price", ascending=False)
    )

    show_all = st.toggle("Show all neighbourhoods", value=False)
    visible_df = neighbourhood_df if show_all else neighbourhood_df.head(10)

    for _, row in visible_df.iterrows():
        st.markdown(
            f"""
            <div class="listing-card">
                <div>
                    <div class="listing-card-title">{row['neighbourhood_cleansed']}</div>
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

# Price vs. Capacity 
with tab4:
    st.markdown('<div class="section-header">Price vs. Capacity</div>', unsafe_allow_html=True)

    selected_room_type_3 = st.selectbox("Room Type", room_types, key="capacity_room_type")

    capacity_df = (
        df[df["room_type"] == selected_room_type_3]
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
