import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Content Monetization Modeler", page_icon="📈", layout="wide" )

with open('best_lr.pkl', 'rb') as file:
    model = pickle.load(file) 
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file) 
with open('features.pkl', 'rb') as file:
    features = pickle.load(file) 

try:
    results = pd.read_csv("model_comparision.csv")
except:
    results = None

st.title("📈 Content Monetization Modeler")

st.header("""
Predict YouTube Ad Revenue using Machine Learning.

This application estimates ad revenue based on video performance metrics,
audience engagement, content category, viewing device, and country.
""")


st.sidebar.header("Video Information")

views = st.sidebar.number_input("Views", min_value=1, value=10000)
likes = st.sidebar.number_input("Likes", min_value=0, value=1000)
comments = st.sidebar.number_input("Comments",min_value=0,value=100)
watch_time = st.sidebar.number_input("Watch Time (Minutes)", min_value=0.0, value=30000.0)
video_length = st.sidebar.number_input("Video Length (Minutes)", min_value=0.1, value=10.0)
subscribers = st.sidebar.number_input("Subscribers", min_value=0, value=100000)

category = st.sidebar.selectbox( "Category", ["Education", "Entertainment", "Gaming", "Lifestyle", "Music", "Tech"])
device = st.sidebar.selectbox( "Device", [ "Desktop", "Mobile", "TV", "Tablet"])

country = st.sidebar.selectbox( "Country",[ "CA", "DE", "IN", "UK", "US"])

if st.sidebar.button("Predict Revenue"):

    input_df = pd.DataFrame(0,index=[0],columns=features)

    input_df["views"] = views
    input_df["likes"] = likes
    input_df["comments"] = comments
    input_df["watch_time_minutes"] = watch_time
    input_df["video_length_minutes"] = video_length
    input_df["subscribers"] = subscribers

    today = datetime.today()

    input_df["year"] = today.year
    input_df["month"] = today.month
    input_df["day"] = today.day
    input_df["weekday"] = today.weekday()

    engagement_rate = (likes + comments) / max(views, 1)
    likes_per_view = (likes) / max(views, 1)
    comments_per_view = (comments) / max(views, 1)
    watch_time_per_view = (watch_time) / max(views, 1)

    input_df["engagement_rate"] = engagement_rate
    input_df["likes_per_view"] = likes_per_view
    input_df["comments_per_view"] = comments_per_view
    input_df["watch_time_per_view"] = watch_time_per_view

    category_col = f"category_{category}"

    if category_col in input_df.columns:
        input_df[category_col] = 1

    device_col = f"device_{device}"

    if device_col in input_df.columns:
        input_df[device_col] = 1

    country_col = f"country_{country}"

    if country_col in input_df.columns:
        input_df[country_col] = 1

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"💰 Predicted Ad Revenue: ${prediction:.2f}")
    st.subheader("Input Summary")
    st.dataframe(input_df)

st.markdown("---")

st.header("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("R² Score","0.9505")
with col2:
    st.metric("MAE","3.25")
with col3:
    st.metric("RMSE","13.78")

# MODEL COMPARISON

if results is not None:

    st.markdown("---")
    st.header("Model Comparison")
    st.dataframe(results, use_container_width=True)

st.markdown("---")

st.header("🔍 Key Insights")

st.markdown("""
### Findings

- Videos with higher total watch time generate significantly more advertising revenue.
- Higher audience engagement tends to accompany higher revenue.
- Linear Regression achieved the highest predictive performance among all tested models.
- Audience engagement positively impacts monetization performance.
- Complex ensemble models such as Random Forest, Gradient Boosting, and XGBoost did not outperform Linear Regression due to the highly linear relationship present in the dataset.
""")


st.markdown("---")

st.caption("Content Monetization Modeler | Machine Learning Regression Project")
