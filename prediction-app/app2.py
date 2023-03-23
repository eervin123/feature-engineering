import streamlit as st
import pandas as pd
import ta
import joblib

# Load the trained models
model_10min = joblib.load('model_10min.pkl')
model_15min = joblib.load('model_15min.pkl')
model_30min = joblib.load('model_30min.pkl')

# Define the function to make predictions
def predict_next_price(data):
    # Add TA features
    data = ta.add_all_ta_features(data, open="open", high="high", low="low", close="close", volume="volume")

    # Use the models to make predictions
    X_new = data.iloc[-1:].values
    price_10min = model_10min.predict(X_new)[0]
    price_15min = model_15min.predict(X_new)[0]
    price_30min = model_30min.predict(X_new)[0]

    return price_10min, price_15min, price_30min

# Define the Streamlit app
st.set_page_config(page_title="Realtime Market Data", layout="wide")
st.title("Realtime Market Data")

# Define the initial data
data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

# Define the loop to update the data and make predictions
while True:
    # Update the data
    new_data = get_new_data()
    data = data.append(new_data, ignore_index=True)

    # Make predictions
    price_10min, price_15min, price_30min = predict_next_price(data)

    # Display the results
    st.write(f"Price in 10 minutes: {price_10min}")
    st.write(f"Price in 15 minutes: {price_15min}")
    st.write(f"Price in 30 minutes: {price_30min}")

    # Wait for 30 seconds
    time.sleep(30)
