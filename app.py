import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------------
#  Load the model and scaler
# ------------------------------------------------------
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------------------------------------------
#  Streamlit UI
# ------------------------------------------------------
st.title("📈 Gold Price Prediction App")
st.write("Enter the values below to predict the Gold Price:")

# User Inputs
SPX = st.number_input("SPX (US stock market (500 Index)",min_value=675, max_value=3325)
USO = st.number_input("USO (Crude Oil Price)",min_value=5,max_value=70)
SLV = st.number_input("SLV (Silver Price)", min_value=5,max_value=45)
EUR_USD = st.number_input("EUR/USD Exchange Rate",min_value=1,max_value=2)

# Predict Button
if st.button("Predict Gold Price"):
    
    # Prepare the input for the model
    input_data = np.array([[SPX, USO, SLV, EUR_USD]])

    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Show result
    st.success(f"💰 Predicted Gold Price: **{prediction[0]:.5f} USD**")
