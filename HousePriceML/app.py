import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("🏠 House Price Prediction App")

# Load data
df = pd.read_csv("housing_prices.csv")
df.dropna(inplace=True)

# Split features & target
X = df.drop("Price", axis=1)
y = df["Price"]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Enter House Details")

input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")

# Model accuracy
st.write("### Model Performance")
st.write("R² Score:", r2_score(y_test, model.predict(X_test)))
