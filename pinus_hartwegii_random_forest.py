import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # for saving the model
import numpy as np  # Add this import

# Load dataset
df = pd.read_csv("pinus_hartwegii_points.csv", encoding='latin-1')
df = df.drop(columns=["Species"])

# Features and target
X = df.drop(columns=["Germination Rate (%)"])
y = df["Germination Rate (%)"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
rf_preds = rf.predict(X_test)
print("Random Forest R2:", r2_score(y_test, rf_preds))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))  # Use np.sqrt instead

# Save model
joblib.dump(rf, "rf_germination_model.pkl")
print("Random Forest model saved!")

# ---------------------------
# Load model and predict new conditions
# Example new input: temp=25, moisture=70, altitude=1500, pH=6.5, light=12
loaded_rf = joblib.load("rf_germination_model.pkl")

new_conditions = [[28, 60, 1500, 6.5, 14]]  # must be a 2D list
prediction = loaded_rf.predict(new_conditions)
print(f"Predicted Germination Rate: {prediction[0]:.2f}%")
