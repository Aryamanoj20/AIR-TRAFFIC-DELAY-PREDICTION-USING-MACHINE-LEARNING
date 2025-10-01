# train_rf_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("📂 Loading dataset...")
df = pd.read_csv(r"C:\Users\user\Downloads\PROJECT\processed_flights_with_weather (1).csv")
print("✅ Dataset loaded. Shape:", df.shape)

print("🛠️ Preprocessing target column ArrDelay...")
df["ArrDelay"] = df["ArrDelay"].apply(lambda x: max(0, x))
print("✅ Target column processed.")

print("📊 Splitting features and target...")
y = df["ArrDelay"]
X = df.drop(["ArrDelay", "Date", "Delayed"], axis=1)

print("🔀 Splitting train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split complete. Train size:", X_train.shape, " Test size:", X_test.shape)

categorical_cols = ["Operating_Airline ", "Origin", "Dest", "airport"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

print("⚙️ Setting up preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)
print("✅ Preprocessor ready.")

print("🌲 Building RandomForest model pipeline...")
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])
print("✅ Model pipeline created.")

print("🚀 Training model...")
rf_model.fit(X_train, y_train)
print("✅ Training complete.")

print("📈 Running predictions on test set...")
y_pred = rf_model.predict(X_test)
print("✅ Predictions complete.")

print("📊 Evaluating model performance...")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

print("💾 Saving trained model...")
joblib.dump(rf_model, "flight_delay_rf_model.pkl")
print("✅ Model saved as flight_delay_rf_model.pkl")
