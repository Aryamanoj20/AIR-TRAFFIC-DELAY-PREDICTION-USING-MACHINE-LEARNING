import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(page_title="Flight Delay Prediction", layout="wide")

# -------------------- BACKGROUND & STYLE -------------------- #
page_bg = """
<style>
/* App background - soft sky gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #d6f0ff, #ffffff); 
}

/* Remove default header background */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Title and text styling */
h1, p, label, .stMarkdown, .css-10trblm, .css-1d391kg {
    color: black !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Card-like box for prediction */
.result-box {
    background-color: rgba(240, 248, 255, 0.9);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- Load model and dataset ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("flight_delay_rf_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("processed_flights_with_weather (1).csv")

model = load_model()
df = load_dataset()

# Dropdown options from dataset
airlines = sorted(df["Operating_Airline "].dropna().unique())
origins = sorted(df["Origin"].dropna().unique())
destinations = sorted(df["Dest"].dropna().unique())

# -------------------- TITLE & DESCRIPTION -------------------- #
st.markdown("<h1 style='text-align: center;'>✈️ Flight Delay Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 0;'>Provide flight and weather details to predict arrival delay in minutes:</p>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:5px; margin-bottom:15px;'>", unsafe_allow_html=True)

# ---------------- Input Widgets in 3 Columns ---------------- #
col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox("Operating Airline", airlines)
    origin = st.selectbox("Origin Airport", origins)
    dest = st.selectbox("Destination Airport", destinations)
    airport = origin  # Assign Weather Airport automatically as Origin

with col2:
    dep_delay = float(st.text_input("Departure Delay (minutes)", value="0.0"))
    wind = float(st.text_input("Windspeed 10m (km/h)", value="10.0"))
    precip = float(st.text_input("Precipitation (mm)", value="0.0"))

with col3:
    temp = float(st.text_input("Temperature 2m (°C)", value="25.0"))
    humidity = float(st.text_input("Relative Humidity 2m (%)", value="50.0"))

# ---------------- Predict Button ---------------- #
if st.button("Predict Delay"):
    input_data = {
        "Operating_Airline ": airline,  # trailing space to match training column
        "Origin": origin,
        "Dest": dest,
        "airport": airport,
        "DepDelay": dep_delay,
        "temperature_2m": temp,
        "windspeed_10m": wind,
        "precipitation": precip,
        "relative_humidity_2m": humidity
    }
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    st.markdown(
        f"<div class='result-box'><h2>⏱️ Predicted Arrival Delay: {prediction:.2f} minutes</h2></div>",
        unsafe_allow_html=True
    )
