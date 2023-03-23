# This will be a streamlit app that will consistently stream the 
# predicted price change for the next 10, 15, 20, 30, and 60 minutes

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os
import numpy as np
from data_funcs import load_data_with_features, load_preprocess_and_save_data
from predict import make_prediction, calculate_and_print_results, print_forecast
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Description of the app
st.title("Bitcoin Price Predictor")
st.write("This app uses machine learning to predict the future price of Bitcoin.")

# Load the data
data = load_data_with_features()

# Show the current price of bitcoin and a plot of the last 30 days
current_price = data['Close'].iloc[-1]
change_24h = data['Close'].iloc[-1] - data['Close'].iloc[-1440]  # 1440 minutes = 24 hours

# Set up the boxes for the current price and 24-hour change
col1, col2 = st.columns(2)
with col1:
    st.write("Current price:")
    if current_price >= data['Close'].iloc[-2]:
        st.write(f"<p style='font-size: 24pt; color: green;'>${current_price:.2f}</p>", unsafe_allow_html=True)
    else:
        st.write(f"<p style='font-size: 24pt; color: red;'>${current_price:.2f}</p>", unsafe_allow_html=True)
with col2:
    st.write("24-hour change:")
    if change_24h >= 0:
        st.write(f"<p style='font-size: 24pt; color: green;'>+${change_24h:.2f}</p>", unsafe_allow_html=True)
    else:
        st.write(f"<p style='font-size: 24pt; color: red;'>-${abs(change_24h):.2f}</p>", unsafe_allow_html=True)
        
# Get the updated model
forecast_periods = [10, 15, 20, 30, 60]
models = []
model_folder_path = 'prediction-app/models'
# Look through the models folder for the model with the correct forecast period
model_files = [f for f in os.listdir(model_folder_path) if f.startswith('rf_model_forecast_')]
model_files = [os.path.join(model_folder_path, f) for f in model_files if int(f.split('_')[3]) in forecast_periods]

logging.info(f"Found {len(model_files)} model files. They are {model_files}")

# Define the options for the dropdown to plot historical hours of bitcoin
options = [1,3,5,24,168] # 168 is 1 week
# Set up the subplots
fig, ax = plt.subplots() # 

# Get the predictions
results_df = pd.DataFrame(columns=["Forecast Period", "y_val", "y_pred"])

# Get the predictions
accuracy_df = pd.DataFrame(columns=[])

# Define the axis for the predictions plot
predictions_fig, predictions_ax = plt.subplots()

for i, forecast_period in enumerate(forecast_periods):
    logging.info(f"Loading model for the {i}'th model with a forecast period of {forecast_period} minutes...")
    logging.info(f"Looking for a model file with the name {model_files[0]}")
    if model_files[i]:
        # Plot the predictions on the same figure
        threshold = 0.001
        y_pred, y_val, price_change = make_prediction(data, forecast_period, model_files[i])
        forecast_price = (y_pred[-1] * data['Close'].iloc[-1]) + (data['Close'].iloc[-1])
        results = calculate_and_print_results(y_val=y_val, y_pred=y_pred, threshold=threshold, forecast_period=forecast_period)
        # results_df = results_df.append({"Forecast Period": f"{forecast_period} minutes", "y_val": y_val, "y_pred": y_pred}, ignore_index=True)
        accuracy_df = accuracy_df.append({"Forecast Period": f"{forecast_period} minutes", 
                                           "Forecast Price Change": f"${price_change:.2f}", 
                                           f"Target Accuracy (+/-{threshold})": f"{results['precision_accuracy']:.4f}%", 
                                           "Direction Accuracy": f"{results['direction_accuracy']:.4f}%"}
                                          , ignore_index=True)
        # Plot the predicted values on the same figure
        predictions_ax.plot(forecast_price, label=f"{forecast_period} minutes")

        time_index = data.index[-1] + pd.Timedelta(minutes=forecast_period)
        ax.scatter(time_index, forecast_price, marker='*', s=100, label=f"{forecast_period} minutes")

st.write("Forecasts: ")
st.write(accuracy_df[['Forecast Period', 'Forecast Price Change']].set_index('Forecast Period'))
st.write("Historical accuracy:")
st.write(accuracy_df[['Forecast Period', f'Target Accuracy (+/-{threshold})', 'Direction Accuracy']].set_index('Forecast Period'))


# Add a dropdown to select the number of days to display
num_hours = st.selectbox('Select the number of hours to display of historical BTC prices', options)
plot_filter = num_hours*60
ax.plot(data.index[-plot_filter:], data['Close'].iloc[-plot_filter:], label='Actual Price') # last num_days days

# Show the actual and predicted values on the same plot
ax.set_title('Actual and Predicted Prices')
ax.set_xlabel('Time')
ax.set_ylabel('Price (USD)')
ax.legend()
st.pyplot(fig)

# st.bar_chart(accuracy_df[['Forecast Period', 'Direction Accuracy', 'Target Accuracy (+/-0.001)']].set_index("Forecast Period"))
def plot_accuracy_chart(accuracy_df):
    accuracy_df = accuracy_df.set_index("Forecast Period")
    accuracy_df['Direction Accuracy'] = accuracy_df['Direction Accuracy'].apply(lambda x: float(x.strip('%')))
    accuracy_df[f'Target Accuracy (+/-0.001)'] = accuracy_df[f'Target Accuracy (+/-0.001)'].apply(lambda x: float(x.strip('%')))

    melted_df = pd.melt(accuracy_df.reset_index(), id_vars="Forecast Period", value_vars=['Direction Accuracy', f'Target Accuracy (+/-0.001)'], var_name="Accuracy Type", value_name="Accuracy (%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Forecast Period', y='Accuracy (%)', hue='Accuracy Type', data=melted_df, ax=ax)

    ax.set_ylim(0, 100)
    ax.set_title("Forecast Accuracy")
    ax.legend()

    st.pyplot(fig)
plot_accuracy_chart(accuracy_df)
# update the data
st.write('Get new data')
st.button('Refresh the Data')
# load_preprocess_and_save_data()
