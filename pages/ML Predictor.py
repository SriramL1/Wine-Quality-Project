# pages/ML Predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Backend_API import load_data

st.title("Wine Quality Predictor")
st.write("Enter wine features to predict its quality (0-10) using a Random Forest model.")

@st.cache_data
def cached_load_data():
    df = load_data()
    return df

with st.spinner("Loading data from BigQuery..."):
    df = cached_load_data()

features = [col for col in df.columns if col != 'quality']
X = df[features]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

st.subheader("Predict Wine Quality")
input_data = {}
for feature in features:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    default_val = float(X[feature].mean())
    # Ensure min_val < max_val
    if min_val == max_val:
        max_val = min_val + 0.1  # Add a small offset to create a valid range
    # Ensure default_val is within the range
    default_val = max(min(default_val, max_val), min_val)
    input_data[feature] = st.slider(f"{feature}", min_val, max_val, default_val)

input_df = pd.DataFrame([input_data], columns=features)

if st.button("Predict Quality"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Wine Quality: {prediction:.1f}")

if st.checkbox("Show Feature Importance"):
    importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    importance = importance.sort_values('Importance', ascending=False)
    st.subheader("Feature Importance")
    st.bar_chart(importance.set_index('Feature'))