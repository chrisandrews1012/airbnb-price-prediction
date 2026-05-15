import streamlit as st

st.set_page_config(
    page_title="Edinburgh Airbnb Intelligence",
    layout="wide",
)

# Inject global CSS
with open("src/airbnb_price_prediction/app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

pages = [
    st.Page("src/airbnb_price_prediction/app/pages/price_predictor.py", title="Price Predictor", default=True),
    st.Page("src/airbnb_price_prediction/app/pages/market_analysis.py", title="Market Analysis"),
    st.Page("src/airbnb_price_prediction/app/pages/neighbourhood_explorer.py", title="Neighbourhood Explorer"),
    st.Page("src/airbnb_price_prediction/app/pages/model_insights.py", title="Model Insights"),
]

pg = st.navigation(pages, position="top")
pg.run()