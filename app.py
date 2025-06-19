import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Load models ====
rf_soil = joblib.load("models/rf_soil.pkl")
rf_yield = joblib.load("models/rf_yield.pkl")

# ==== Page setup ====
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield & Fertilizer Recommendation App")

# ==== Tabs ====
tab1, tab2, tab3 = st.tabs(["🔬 Soil & Yield Prediction", "🌱 Fertilizer Suggestion", "🌦 Weather Info"])

with tab1:
    st.header("🧪 Predict Soil Organic Matter")
    pH = st.slider("pH", 5.0, 8.0, 6.5)
    N = st.slider("Nitrogen (N)", 0, 100, 50)
    P = st.slider("Phosphorus (P)", 0, 100, 40)
    K = st.slider("Potassium (K)", 0, 100, 30)

    soil_features = np.array([[pH, N, P, K]])
    organic_matter_pred = rf_soil.predict(soil_features)[0]
    st.success(f"Predicted Organic Matter: {organic_matter_pred:.2f}")

    st.divider()
    st.header("🌾 Predict Crop Yield")
    temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
    humidity = st.slider("Humidity (%)", 20.0, 100.0, 60.0)
    rainfall = st.slider("Rainfall (mm)", 10.0, 300.0, 100.0)

    yield_features = np.array([[N, P, K, temperature, humidity, rainfall]])
    predicted_yield = rf_yield.predict(yield_features)[0]
    st.success(f"Predicted Crop Yield: {predicted_yield:.2f} Q/Ha")

with tab2:
    st.header("🌿 Fertilizer Recommendation")
    crop_input = st.text_input("Enter Crop Name (e.g., Wheat)")
    soil_input = st.text_input("Enter Soil Type (e.g., Loamy)")

    # Static fertilizer data (can be loaded from CSV too)
    fertilizer_df = pd.DataFrame({
        'Crop': ['Wheat', 'Rice', 'Corn'],
        'SoilType': ['Loamy', 'Clay', 'Sandy'],
        'Fertilizer': ['NPK 10-26-26', 'Urea', 'DAP'],
        'Dosage': ['250 kg/ha', '175 kg/ha', '200 kg/ha']
    })

    if st.button("Recommend Fertilizer"):
        row = fertilizer_df[
            (fertilizer_df["Crop"].str.lower() == crop_input.lower()) &
            (fertilizer_df["SoilType"].str.lower() == soil_input.lower())
        ]
        if not row.empty:
            fert = row.iloc[0]["Fertilizer"]
            dose = row.iloc[0]["Dosage"]
            st.success(f"Recommended Fertilizer: {fert}, Dosage: {dose}")
        else:
            st.warning("No match found. Try with valid crop & soil.")

with tab3:
    st.header("☁️ Live Weather Forecast")
    city = st.text_input("Enter City Name (e.g., Nashik)")
    api_key = st.text_input("Enter your OpenWeatherMap API Key", type="password")

    def get_weather(city, api_key):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url).json()
        if res.get("main"):
            return {
                "Temperature (°C)": res["main"]["temp"],
                "Humidity (%)": res["main"]["humidity"],
                "Pressure (hPa)": res["main"]["pressure"],
                "Description": res["weather"][0]["description"]
            }
        else:
            return {"Error": res.get("message", "Unknown error")}

    if st.button("Get Weather"):
        weather = get_weather(city, api_key)
        for k, v in weather.items():
            st.write(f"**{k}:** {v}")
