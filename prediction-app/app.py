# This will be a streamlit app that will consistently stream the 
# predicted price change for the next 10, 15, 20, 30, and 60 minutes

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os
import numpy as np
from data import load_data_with_features
from predict import make_prediction, calculate_and_print_results, print_forecast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Description of the app
st.title("Bitcoin Price Predictor")
st.write("This app uses machine learning to predict the future price of Bitcoin.")

# Load the data
data = load_data_with_features()

# Show the current price of bitcoin and a plot of the last 30 days
current_price = data['Close'].iloc[-1]
change_24h = data['Close'].iloc[-1] - data['Close'].iloc[-1440]  # 1440 minutes = 24 hours
st.write(f"Current price: ${current_price:.2f}")
st.write(f"Change in the last 24 hours: ${change_24h:.2f}")
st.write("30-day price history:")
fig, ax = plt.subplots()
sns.lineplot(data=data['Close'].tail(43200))  # 43200 minutes = 30 days
st.pyplot(fig)

# Get the updated model
forecast_periods = [10, 15, 20, 30, 60]
models = []
model_folder_path = 'prediction-app/models'
# Look through the models folder for the model with the correct forecast period
model_files = [f for f in os.listdir(model_folder_path) if f.startswith('rf_model_forecast_')]
model_files = [os.path.join(model_folder_path, f) for f in model_files if int(f.split('_')[3]) in forecast_periods]

logging.info(f"Found {len(model_files)} model files. They are {model_files}")

# Define the options for the dropdown
options = [1,2,3,4,5,6]

# Add a dropdown to select the number of days to display
num_hours = st.selectbox('Select the number of hours to display', options)

# Update the plot_filter variable based on the selected option
plot_filter = num_hours * 60

# Plot the data using the updated plot_filter variable
# Plot the data using the updated plot_filter variable
fig, ax = plt.subplots()
ax.plot(data.index[-plot_filter:], data['Close'].iloc[-plot_filter:], label='Actual Price') # last num_days days

# Get the predictions
results_df = pd.DataFrame(columns=["Forecast Period", "y_val", "y_pred"])

# Get the predictions
accuracy_df = pd.DataFrame(columns=["Forecast Period", "Price Change", "Accuracy", "Direction Accuracy"])
# Define the axis for the predictions plot
predictions_fig, predictions_ax = plt.subplots()

for i, forecast_period in enumerate(forecast_periods):
    logging.info(f"Loading model for the {i}'th model with a forecast period of {forecast_period} minutes...")
    logging.info(f"Looking for a model file with the name {model_files[0]}")
    if model_files[i]:
        # Plot the predictions on the same figure

        y_pred, y_val, price_change = make_prediction(data, forecast_period, model_files[i])
        forecast_price = (y_pred[-1] * data['Close'].iloc[-1]) + (data['Close'].iloc[-1])
        results = calculate_and_print_results(y_val=y_val, y_pred=y_pred, threshold=0.001, forecast_period=forecast_period)
        # results_df = results_df.append({"Forecast Period": f"{forecast_period} minutes", "y_val": y_val, "y_pred": y_pred}, ignore_index=True)
        accuracy_df = accuracy_df.append({"Forecast Period": f"{forecast_period} minutes", 
                                           "Price Change": f"${price_change:.2f}", 
                                           "Accuracy": f"{results['accuracy']:.4f}%", 
                                           "Direction Accuracy": f"{results['direction_accuracy']:.4f}%"}
                                          , ignore_index=True)
        # Plot the predicted values on the same figure
        predictions_ax.plot(forecast_price, label=f"{forecast_period} minutes")

        time_index = data.index[-1] + pd.Timedelta(minutes=forecast_period)
        ax.scatter(time_index, forecast_price, marker='*', s=100, label=f"{forecast_period} minutes")

# Show the actual and predicted values on the same plot
ax.set_title('Actual and Predicted Prices')
ax.set_xlabel('Time')
ax.set_ylabel('Price (USD)')
ax.legend()
st.pyplot(fig)

st.write("Historical accuracy:")
st.write(accuracy_df)
st.bar_chart(accuracy_df.set_index("Forecast Period"))
