import pandas as pd
import joblib

print("📂 Loading trained model...")
model = joblib.load("flight_delay_rf_model.pkl")
print("✅ Model loaded successfully.")

# Function to get user input
def get_user_input():
    print("\n📝 Enter flight details for prediction:\n")
    airline = input("Operating Airline (e.g., WN): ")
    origin = input("Origin Airport (e.g., PHX): ")
    dest = input("Destination Airport (e.g., IAH): ")
    airport = input("Weather Airport (e.g., PHX): ")
    dep_delay = float(input("Departure Delay (in minutes): "))

    temp = float(input("Temperature 2m (°C): "))
    wind = float(input("Windspeed 10m (km/h): "))
    precip = float(input("Precipitation (mm): "))
    humidity = float(input("Relative Humidity 2m (%): "))

    print("✅ User input collected.")
    return {
        "Operating_Airline ": airline,  # Note: trailing space kept to match training
        "Origin": origin,
        "Dest": dest,
        "airport": airport,
        "DepDelay": dep_delay,
        "temperature_2m": temp,
        "windspeed_10m": wind,
        "precipitation": precip,
        "relative_humidity_2m": humidity
    }

print("📝 Collecting input from user...")
user_data = get_user_input()
input_df = pd.DataFrame([user_data])
print("✅ Input data prepared for prediction.")

print("🔮 Making prediction...")
pred_delay = model.predict(input_df)[0]
print(f"\n✈️ Predicted flight delay: {pred_delay:.2f} minutes")
