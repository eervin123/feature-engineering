# This will be a streamlit app that will consistently stream the 
# predicted price change for the next 10, 15, 20, 30, and 60 minutes

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os
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
options = [1, 5, 7, 14, 30]

# Add a dropdown to select the number of hours to display
num_hours = st.selectbox('Select the number of hours to display', options)

# Update the plot_filter variable based on the selected option
plot_filter = num_hours * 60

# Plot the data using the updated plot_filter variable
fig, ax = plt.subplots()
ax.plot(data.index[-plot_filter:], data['Close'].iloc[-plot_filter:], label='Actual Price') # last num_hours hours

# Get the predictions
accuracy_df = pd.DataFrame(columns=["Forecast Period", "Accuracy", "MSE", "RMSE", "Precision Accuracy", "Direction Accuracy", "R-squared"])
for i, forecast_period in enumerate(forecast_periods):
    logging.info(f"Loading model for the {i}'th model with a forecast period of {forecast_period} minutes...")
    logging.info(f"Looking for a model file with the name {model_files[0]}")
    st.write(f"Predicted price change in {forecast_period} minutes:")
    if model_files[i]:
        y_pred, y_val, price_change = make_prediction(data, forecast_period, model_files[i])
        st.write(f"The current price of bitcoin is {current_price}. Predicted price change in the next {forecast_period} minutes is: ${price_change:.2f}")
        fig, ax = plt.subplots()
        ax.plot(data.index[-plot_filter:], data['Close'].iloc[-plot_filter:], label='Actual Price') # last num_hours hours
        ax.plot(data.index[-1:], data['Close'].iloc[-1:], marker='o', markersize=5, label='Current Price')
        ax.plot(data.index[-1:] + pd.Timedelta(minutes=forecast_period), price_change + current_price, marker='*', markersize=10, label='Predicted Price')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price ($)')
        ax.legend()
        st.pyplot(fig)
        results = calculate_and_print_results(y_val=y_val, y_pred=y_pred, threshold=0.001, forecast_period=forecast_period)
        accuracy_df = pd.concat([accuracy_df, pd.DataFrame({"Forecast Period": f"{forecast_period} minutes", "Accuracy": results['accuracy'], "MSE": results['mse'], "RMSE": results['rmse'], "Precision Accuracy": results['accuracy'], "Direction Accuracy": results['direction_accuracy'], "R-squared": results['r2']}, index=[0])])
        st.write("Prediction statistics:")
        st.table(pd.DataFrame.from_dict(results, orient='index').T)
    else:
        st.write("Model not available.")

# Show historical accuracy and forecasts        
st.write("Historical accuracy:")
st.write(accuracy_df)
st.bar_chart(accuracy_df.set_index("Forecast Period"))



